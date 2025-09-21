class RedForceController:
    """
    精简版 RedForceController —— 已移除：
      * 导弹躲避（Missile Evasion）逻辑及相关状态/计时/打印
      * “特殊飞机（third）发射后立刻反向拉满速”逻辑及相关状态/标志
    保留功能：
      - 雷达启用/看门狗
      - 初始速度设定与后续 BOOST（BOOST_AFTER_SEC）
      - 目标分配、fire 请求、fire 成功确认与 ammo 管理
      - destroyed 状态更新（get_situ_info）
      - 每秒观测向量采集与 CSV 落盘（via RLRecorder）
      - run_loop 与优雅结束
    注意：如果你的上层或其它代码依赖这些被删的字典/字段（例如 _evasive_until、_missile_prev_xy_t 等），请一并移除对它们的引用；本类内部已不再使用它们。
    """

    def _radar_key_present(self, rid: int) -> bool:
        try:
            vis_raw = self.client.get_visible_vehicles() or {}
            keys = set()
            for k in vis_raw.keys():
                try:
                    keys.add(int(k))
                except Exception:
                    pass
            return int(rid) in keys
        except Exception:
            return False

    def _enable_radar_reliably(self, rid: int, retries: int = 3, gap: float = 0.2) -> bool:
        for i in range(retries):
            try:
                uid = self.client.enable_radar(vehicle_id=rid, state=1)
                print(f"[Red] enable_radar({rid},1) uid={uid}", flush=True)
            except Exception as e:
                print(f"[Red] enable_radar({rid},1) error: {e}", flush=True)
            time.sleep(gap)

            if self._radar_key_present(rid):
                print(f"[Red] Radar ON confirmed for {rid}", flush=True)
                return True

            if i == 0:
                try:
                    self.client.enable_radar(vehicle_id=rid, state=0)
                    print(f"[Red] enable_radar({rid},0) (toggle)", flush=True)
                except Exception as e:
                    print(f"[Red] enable_radar({rid},0) error: {e}", flush=True)
                time.sleep(0.1)

        print(f"[Red] Radar still OFF after retries for {rid}", flush=True)
        return False

    def _radar_watchdog_once(self):
        time.sleep(2.0)
        try:
            for rid in self.red_ids:
                if not self._radar_key_present(int(rid)):
                    print(f"[RadarWatchdog] rid={rid} seems OFF; re-enable...", flush=True)
                    self._enable_radar_reliably(int(rid), retries=3, gap=0.2)
        except Exception as e:
            print("[RadarWatchdog] check failed:", e, flush=True)

    def __init__(self, client, red_ids, out_csv_path):
        self.client = client
        self.red_ids_seq = [int(x) for x in red_ids]
        self.red_ids = set(self.red_ids_seq)
        self.ammo = {int(r): AMMO_MAX for r in red_ids}
        self.last_fire_time = {int(r): 0.0 for r in red_ids}
        self.assigned_target = {}
        self.target_in_progress = set()
        self.destroyed_targets = set()
        self.destroyed_blue = set()
        self.destroyed_red = set()
        self._ended = False
        self.lock = threading.Lock()
        self.target_locks = {}
        self._last_visible = set()
        self._last_locked = {}
        self._last_destroyed = set()
        self.recorder = RLRecorder()
        self._last_logged_sec = -1
        self._score_cache = None
        self._score_counter = 0
        self.out_csv_path = out_csv_path

        # Debug printing for missile internals removed to avoid relying on evasion code.

        # 初始化雷达
        for rid in red_ids:
            try:
                uid = self.client.enable_radar(vehicle_id=rid, state=1)
                print(f"[Red] Radar ON for {rid}, uid={uid}", flush=True)
            except Exception as e:
                print(f"[Red] enable_radar({rid}) failed: {e}", flush=True)

        # 开局定速向东（保持你原始的初始速度设定）
        try:
            t0 = self.client.get_sim_time()
        except Exception:
            t0 = 0.0

        for rid in sorted(self.red_ids):
            spd = float(INITIAL_SPEEDS.get(int(rid), 100.0))
            vcmd = rflysim.Vel()
            vcmd.rate = spd
            vcmd.direct = EAST_HEADING_DEG
            vcmd.vz = vel_0.vz
            try:
                uid = self.client.set_vehicle_vel(int(rid), vcmd)
                self.recorder.mark_vel_cmd(int(rid), rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=t0)
                print(f"[INIT-SPEED] rid={rid} -> speed={vcmd.rate} heading={vcmd.direct}° vz={vcmd.vz}", flush=True)
            except Exception as e:
                print(f"[INIT-SPEED] set_vehicle_vel({rid}) failed: {e}", flush=True)

        self._init_speed_done = True
        self._boost_done = {rid: False for rid in self.red_ids}
        threading.Thread(target=self._radar_watchdog_once, daemon=True).start()

        # removed special-plane marker / special burn flags entirely (no _rid_third_sorted, no _did_special_afterlock_burn)

    # ---- 内部工具（保留） ----
    def _estimate_msl_heading_speed(self, mid, lon_now, lat_now, t_now):
        prev = getattr(self, "_missile_last", {}).get(int(mid))
        if not hasattr(self, "_missile_last"):
            self._missile_last = {}
        self._missile_last[int(mid)] = {"lon": float(lon_now), "lat": float(lat_now), "t": float(t_now)}
        if not prev:
            return (None, None)

        dt = float(t_now) - float(prev["t"])
        if dt <= 1e-3:
            return (None, None)

        mdir = _bearing_deg_from_A_to_B(prev["lon"], prev["lat"], lon_now, lat_now)
        dist = _geo_dist_haversine_m(prev["lon"], prev["lat"], lon_now, lat_now) or 0.0
        mspeed = dist / dt
        return (mdir, mspeed)

    def _is_blue_side(self, side):
        return side in (2, "SIDE_BLUE", "BLUE")

    def _is_destroyed_flag(self, dmg):
        if isinstance(dmg, (int, float)):
            return int(dmg) == 1
        s = str(dmg)
        return ("DESTROYED" in s.upper()) or (s.strip() == "1")

    def _is_fire_success(self, uid, max_tries=5, wait_s=0.1):
        if not uid:
            return False
        for _ in range(max_tries):
            time.sleep(wait_s)
            try:
                st = self.client.get_command_status(uid)
            except Exception:
                st = None
            if isinstance(st, dict):
                status = st.get('status', '')
                result = st.get('execute_result', '') or ''
                if status == 'EXECUTE_SUCCESS' or ('执行打击成功' in result):
                    return True
                if status == 'EXECUTE_FAILED':
                    return False
        return False

    def _fire_with_log(self, rid, tid):
        uid = self.client.set_target(vehicle_id=rid, target_id=tid)
        sim_t = self.client.get_sim_time()
        self.recorder.mark_action(rid, {"type": "fire", "target_id": int(tid), "uid": uid}, sim_t)
        return uid

    def _distance_m(self, lon1, lat1, lon2, lat2):
        d = _geo_dist_haversine_m(lon1, lat1, lon2, lat2)
        return float(d) if d is not None else 1e18

    def _get_target_pos(self, target_id, vis, all_pos):
        p = all_pos.get(target_id)
        if p is not None:
            return p.x, p.y
        for tracks in vis.values():
            for t in tracks:
                if t.get("target_id") == target_id:
                    return t.get("lon"), t.get("lat")
        return (None, None)

    def _fetch_score_with_retry(self, tries=1, wait=0.2, where="(unknown)"):
        score_obj = None
        for _ in range(tries):
            try:
                score_obj = self.client.get_score()
            except Exception:
                score_obj = None
            if score_obj:
                print(f"[Final Score {where}]", score_obj, flush=True)
                return score_obj
            time.sleep(wait)
        print(f"[Final Score {where}] still None after retry.", flush=True)
        return None

    def _end_simulation_once(self, reason=""):
        if self._ended:
            return
        self._ended = True
        if reason:
            print(f"[AutoStop] {reason}", flush=True)
        self._fetch_score_with_retry(tries=5, wait=0.2, where="pre-stop")
        try:
            self.client.stop()
        except Exception as e:
            print("[AutoStop] client.stop() failed:", e, flush=True)
        self._fetch_score_with_retry(tries=1, wait=0.2, where="post-stop")
        try:
            self.recorder.dump_csv(self.out_csv_path)
        except Exception as e:
            print("[RL] dump_csv failed:", e, flush=True)

    # ---- 删除的特殊机动函数：_maybe_back_fullspeed_for_special() 已完全移除 ----

    def _update_destroyed_from_situ(self, force=False):
        if self._ended:
            return
        if not force:
            self._situ_counter = getattr(self, "_situ_counter", 0) + 1
            if self._situ_counter % 5 != 0:
                return
        try:
            situ_raw = self.client.get_situ_info() or {}
        except Exception:
            return
        any_new = False
        for vid, info in situ_raw.items():
            if not info:
                continue
            side = getattr(info, "side", None) if not isinstance(info, dict) else info.get("side")
            dmg  = getattr(info, "damage_state", None) if not isinstance(info, dict) else info.get("damage_state")
            try:
                vvid = getattr(info, "id", None)
                if vvid is None and isinstance(info, dict):
                    vvid = info.get("id")
                if vvid is None:
                    vvid = int(vid)
                else:
                    vvid = int(vvid)
            except Exception:
                continue
            if not self._is_destroyed_flag(dmg):
                continue
            if self._is_blue_side(side):
                if vvid not in self.destroyed_blue:
                    self.destroyed_blue.add(vvid); self.destroyed_targets.add(vvid); any_new = True
            else:
                if vvid not in self.destroyed_red:
                    self.destroyed_red.add(vvid); any_new = True
                    if vvid in self.ammo:
                        self.ammo[vvid] = 0
                    self.target_locks.pop(vvid, None)
        if any_new:
            print(f"[Situ] BLUE destroyed: {sorted(self.destroyed_blue)} | RED destroyed: {sorted(self.destroyed_red)}", flush=True)

        import time as _time
        now_wall = _time.time()
        if (len(self.destroyed_blue) >= 4 or len(self.destroyed_red) >= 4) and getattr(self, "_end_grace_until", None) is None:
            if len(self.destroyed_blue) >= 4:
                self._end_reason = "All 4 BLUE aircraft destroyed."
            else:
                self._end_reason = "All 4 RED aircraft destroyed."
            self._end_grace_until = now_wall + 1.0
            print(f"[Grace] {self._end_reason} Ending in 20s...", flush=True)
            return

        if getattr(self, "_end_grace_until", None) is not None:
            if now_wall >= self._end_grace_until:
                self._end_simulation_once(self._end_reason)

    # ---- 主步：采样 + 攻击分配（已移除导弹躲避相关处理） ----
    def step(self):
        if self._ended:
            return
        try:
            raw_visible = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True); return
        vis = normalize_visible(raw_visible)
        try:
            all_pos = self.client.get_vehicle_pos()
        except Exception:
            all_pos = {}
        try:
            vel_all = self.client.get_vehicle_vel()
        except Exception:
            vel_all = {}

        self._update_destroyed_from_situ()
        sim_t = self.client.get_sim_time()
        sim_sec = int(sim_t)

        # BOOST: 到点将其他飞机拉到全速（保持向东）
        if self._init_speed_done and (sim_t >= BOOST_AFTER_SEC):
            try:
                for rid in sorted(self.red_ids):
                    if int(rid) == 10085:
                        self._boost_done[rid] = True
                        continue
                    if self._boost_done.get(rid, False):
                        continue

                    vcmd = rflysim.Vel()
                    vcmd.rate = float(FULL_SPEED)
                    vcmd.direct = EAST_HEADING_DEG
                    try:
                        v_meas = self.client.get_vehicle_vel().get(int(rid))
                        vcmd.vz = getattr(v_meas, "vz", vel_0.vz)
                    except Exception:
                        vcmd.vz = vel_0.vz

                    try:
                        uid = self.client.set_vehicle_vel(int(rid), vcmd)
                        self.recorder.mark_vel_cmd(int(rid), rate=vcmd.rate, direct=vcmd.direct, vz=vcmd.vz, sim_time=sim_t)
                        self._boost_done[rid] = True
                        print(f"[BOOST] t={sim_t:.1f}s rid={rid} -> speed={vcmd.rate} heading={vcmd.direct}° vz={vcmd.vz}", flush=True)
                    except Exception as e:
                        print(f"[BOOST] set_vehicle_vel({rid}) failed: {e}", flush=True)
            except Exception as e:
                print("[BOOST] block failed:", e, flush=True)

        self._score_counter = getattr(self, "_score_counter", 0) + 1
        if self._score_counter % 5 == 0:
            try:
                self._score_cache = self.client.get_score() or None
            except Exception:
                self._score_cache = None

        # 每秒采集 obs/act 并记录
        if sim_sec != getattr(self, "_last_logged_sec", -1):
            try:
                situ_raw_full = self.client.get_situ_info() or {}
            except Exception:
                situ_raw_full = {}

            # 向量/导弹相关简化：仅收集 missiles 的位置用于 nearest_msl_dist（不做躲避决策）
            missiles_pos = {}
            for vid, p in (all_pos or {}).items():
                try:
                    _id = int(vid)
                    if _id < 10000 and getattr(p, "x", None) is not None and getattr(p, "y", None) is not None:
                        missiles_pos[_id] = (float(p.x), float(p.y))
                except Exception:
                    pass
            for tracks in (vis or {}).values():
                for t in (tracks or []):
                    tid = t.get("target_id")
                    if tid is None or int(tid) >= 10000:
                        continue
                    if tid not in missiles_pos:
                        lonB, latB = t.get("lon"), t.get("lat")
                        if lonB is not None and latB is not None:
                            missiles_pos[int(tid)] = (float(lonB), float(latB))

            obs_concat = []; act_concat = []

            for rid in sorted(self.red_ids):
                my_p = all_pos.get(rid)
                my_lon = getattr(my_p, "x", None) if my_p else None
                my_lat = getattr(my_p, "y", None) if my_p else None
                tracks = vis.get(rid, []) or []

                # 最近4个蓝机距离
                dlist = []
                for t in tracks:
                    tid = t.get("target_id")
                    if tid is None or int(tid) < 10000:
                        continue
                    lonB, latB = t.get("lon"), t.get("lat")
                    if None in (lonB, latB, my_lon, my_lat):
                        continue
                    try:
                        d = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0), Position(x=lonB, y=latB, z=0))
                    except Exception:
                        d = _geo_dist_haversine_m(my_lon, my_lat, lonB, latB)
                    if d is not None:
                        dlist.append(float(d))
                dlist.sort()
                if len(dlist) < 4:
                    dlist += [BLUE_DIST_CAP] * (4 - len(dlist))
                else:
                    dlist = dlist[:4]

                # 最近蓝机（速度、方向、方位）
                nb_speed, nb_dir, bearing_red_to_blue = None, None, None
                if tracks and my_lon is not None and my_lat is not None:
                    best, best_t = None, None
                    for t in tracks:
                        tid = t.get("target_id")
                        if tid is None or int(tid) < 10000:
                            continue
                        lonB, latB = t.get("lon"), t.get("lat")
                        if lonB is None or latB is None:
                            continue
                        try:
                            dd = self.client.get_distance_by_lon_lat(Position(x=my_lon, y=my_lat, z=0), Position(x=lonB, y=latB, z=0))
                        except Exception:
                            dd = _geo_dist_haversine_m(my_lon, my_lat, lonB, latB)
                        if dd is not None and (best is None or dd < best):
                            best, best_t = dd, t
                    if best_t:
                        nb_speed = best_t.get("speed")
                        nb_dir = best_t.get("direction")
                        lonB, latB = best_t.get("lon"), best_t.get("lat")
                        if lonB is not None and latB is not None and my_lon is not None and my_lat is not None:
                            bearing_red_to_blue = _bearing_deg_from_A_to_B(my_lon, my_lat, lonB, latB)
                        if nb_dir is None and lonB is not None and latB is not None:
                            nb_dir = _bearing_deg_from_A_to_B(my_lon, my_lat, lonB, latB)

                # 最近导弹（仅计算距离与方位，不触发躲避）
                nearest_msl_dist, nearest_msl_dir, bearing_red_to_msl = None, None, None
                if my_lon is not None and my_lat is not None and missiles_pos:
                    best_mid, best_d, best_pos = None, None, None
                    for mid, (mlon, mlat) in missiles_pos.items():
                        dd = _geo_dist_haversine_m(my_lon, my_lat, mlon, mlat)
                        if dd is None:
                            continue
                        if best_d is None or dd < best_d:
                            best_mid, best_d, best_pos = mid, float(dd), (mlon, mlat)
                    if best_mid is not None:
                        nearest_msl_dist = best_d
                        bearing_red_to_msl = _bearing_deg_from_A_to_B(my_lon, my_lat, best_pos[0], best_pos[1])
                        # 不再尝试估计 nearest_msl_dir（没必要用于当前目标分配）

                jam_signed = _signed_dist_to_jam_boundary_m(self.client, my_lon, my_lat)
                vel_meas = vel_all.get(rid)
                obs_raw = {
                    "pos": {"lon": my_lon, "lat": my_lat, "alt": getattr(my_p, "z", None) if my_p else None},
                    "vel": {"vx": getattr(vel_meas, "vx", None) if vel_meas else None,
                            "vy": getattr(vel_meas, "vy", None) if vel_meas else None,
                            "vz": getattr(vel_meas, "vz", None) if vel_meas else None},
                    "ammo": int(self.ammo.get(rid, 0)),
                }

                obs19 = self.recorder.pack_single_obs19(
                    self.client, rid, obs_raw, BOUNDARY_RECT, jam_signed,
                    dlist, nb_speed, nb_dir, bearing_red_to_blue,
                    nearest_msl_dist, None, bearing_red_to_msl,
                    vel_meas=vel_meas
                )
                obs_concat.extend(obs19)

                act4 = self.recorder.pack_single_act4(rid, sim_sec, vel_meas=vel_meas, pos_meas=my_p)
                act_concat.extend(act4)

                try:
                    from math import sqrt
                    speed_scalar = sqrt(float(obs_raw["vel"]["vx"] or 0.0) ** 2 + float(obs_raw["vel"]["vy"] or 0.0) ** 2)
                except Exception:
                    speed_scalar = None

                self.recorder.record_tick(sim_sec, rid, {
                    "boundary_dists": {"left": obs19[0], "right": obs19[1], "down": obs19[2], "up": obs19[3]},
                    "vel": {"vx": obs19[4], "vy": obs19[5], "vz": obs19[6]},
                    "blue_dists": dlist,
                    "nearest_blue_speed": nb_speed,
                    "nearest_blue_dir": nb_dir,
                    "bearing_to_nearest_blue": bearing_red_to_blue,
                    "nearest_missile_dist": nearest_msl_dist,
                    "nearest_missile_dir": None,
                    "bearing_to_nearest_missile": bearing_red_to_msl,
                    "jam_signed_dist": obs19[17],
                    "ammo": obs19[18],
                    "speed_scalar": speed_scalar,
                })

            self.recorder.latest_obs_vec = obs_concat
            self.recorder.latest_act_vec = act_concat
            self.recorder.add_vector_row(sim_sec, obs_concat, act_concat)
            self._last_logged_sec = sim_sec

        if self._ended:
            return

        # 攻击目标搜集与分配（同原逻辑）
        try:
            raw_visible2 = self.client.get_visible_vehicles()
        except Exception as e:
            print("[Red] get_visible_vehicles() failed:", e, flush=True)
            return
        vis2 = normalize_visible(raw_visible2)
        try:
            all_pos2 = self.client.get_vehicle_pos()
        except Exception:
            all_pos2 = {}
        self._update_destroyed_from_situ()
        now = self.client.get_sim_time()

        try:
            vel_all2 = self.client.get_vehicle_vel()
        except Exception:
            vel_all2 = {}

        visible_blue_targets = set()
        for tracks in vis2.values():
            for t in tracks:
                tid = t.get("target_id")
                if tid is None or tid < 10000:
                    continue
                if tid in self.destroyed_targets or tid in self.target_locks:
                    continue
                visible_blue_targets.add(tid)

        if visible_blue_targets != self._last_visible:
            print("未锁定的目标:", visible_blue_targets, flush=True)
            self._last_visible = set(visible_blue_targets)
        if self.target_locks != self._last_locked:
            print("已经锁定的目标:", self.target_locks, flush=True)
            self._last_locked = dict(self.target_locks)
        if self.destroyed_targets != self._last_destroyed:
            print("已经被摧毁的目标:", self.destroyed_targets, flush=True)
            self._last_destroyed = set(self.destroyed_targets)
        if not visible_blue_targets:
            return

        used_reds_this_round = set()
        assignments = []
        for tid in visible_blue_targets:
            t_lon, t_lat = self._get_target_pos(tid, vis2, all_pos2)
            if t_lon is None or t_lat is None:
                continue
            best_red, best_d = None, 1e18
            for rid in self.red_ids:
                if rid in used_reds_this_round:
                    continue
                if rid in self.destroyed_targets:
                    continue
                if self.ammo.get(rid, 0) <= 0:
                    continue
                if (now - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                    continue
                r_pos = all_pos2.get(rid)
                if r_pos is None:
                    continue
                d = self._distance_m(r_pos.x, r_pos.y, t_lon, t_lat)
                if d < best_d:
                    best_d, best_red = d, rid
            if best_red is not None:
                assignments.append((best_red, tid))
                used_reds_this_round.add(best_red)

        if not assignments:
            return

        for rid, tid in assignments:
            if tid in self.destroyed_targets:
                continue
            now_sim = self.client.get_sim_time()
            if (now_sim - self.last_fire_time.get(rid, 0.0)) < ATTACK_COOLDOWN_SEC:
                continue
            if self.ammo.get(rid, 0) <= 0:
                continue
            try:
                uid = self._fire_with_log(rid, tid)
                print(f"[Red] {rid} -> fire at {tid}, uid={uid}", flush=True)
                ok = self._is_fire_success(uid, max_tries=5, wait_s=0.1)
                if not ok:
                    print(f"[Red] fire NOT confirmed for {rid}->{tid}, keep target available.", flush=True)
                    time.sleep(0.1)
                    continue
                self.recorder.mark_attack_success(rid, self.client.get_sim_time())
                self.ammo[rid] -= 1
                self.last_fire_time[rid] = self.client.get_sim_time()
                self.assigned_target[rid] = tid
                self.target_locks[tid] = {"red_id": rid, "until": self.client.get_sim_time() + LOCK_SEC}
                time.sleep(0.1)
            except Exception as e:
                print(f"[Red] set_target({rid},{tid}) failed: {e}", flush=True)
                continue

    def run_loop(self, stop_when_paused=False, max_wall_time_sec=None):
        start_t = time.time()
        try:
            while True:
                if self._ended:
                    break
                if stop_when_paused:
                    try:
                        if self.client.is_pause():
                            break
                    except Exception:
                        pass
                if max_wall_time_sec is not None and (time.time() - start_t) > max_wall_time_sec:
                    print("[Red] Episode timeout reached, stopping...", flush=True)
                    self._end_simulation_once("Timeout")
                    break
                self.step()
                time.sleep(SCAN_INTERVAL_SEC)
        finally:
            try:
                self.recorder.dump_csv(self.out_csv_path)
            except Exception as e:
                print("[RL] dump_csv on finally failed:", e, flush=True)
