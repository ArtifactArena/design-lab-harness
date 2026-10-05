"""Baseline car controller with aggressive skid-steer point navigation."""

import math


_WHEEL_NAMES = ("motor_fl", "motor_fr", "motor_rl", "motor_rr")
_ZERO_WHEEL_MAP = {name: 0.0 for name in _WHEEL_NAMES}

_WHEEL_RADIUS = 0.16
_TRACK_HALF = 0.30
_MAX_WHEEL_LINEAR = 6.2
_MAX_WHEEL_OMEGA = _MAX_WHEEL_LINEAR / _WHEEL_RADIUS

_LAST_T = None
_LAST_ACTIONS = dict(_ZERO_WHEEL_MAP)
_LAST_LINEAR_CMD = 0.0
_LAST_YAW_CMD = 0.0
_DEADLOCK_COUNT = 0
_BOOST_UNTIL = 0
_SLIP_INT = 0.0
_WHEEL_INT = dict(_ZERO_WHEEL_MAP)
_WHEEL_LAST_ERR = dict(_ZERO_WHEEL_MAP)
_WHEEL_LAST_T = None
_EDGE_RECOVERY_LATCH = False
_ORBIT_DIRECTION = 0
_ORBIT_DIRECTION_LOCK_T = -10**9
_OFFENSE_PHASE = "idle"
_OFFENSE_PHASE_START_T = -10**9
_OFFENSE_RAM_START_EDGE = None
_OFFENSE_ATTEMPT = 0
_OFFENSE_COOLDOWN_UNTIL = -10**9
_POLICY_DEBUG = {}


def _reset_controller_state():
    global _LAST_T, _LAST_ACTIONS, _LAST_LINEAR_CMD, _LAST_YAW_CMD
    global _DEADLOCK_COUNT, _BOOST_UNTIL, _SLIP_INT
    global _WHEEL_INT, _WHEEL_LAST_ERR, _WHEEL_LAST_T, _EDGE_RECOVERY_LATCH
    global _ORBIT_DIRECTION, _ORBIT_DIRECTION_LOCK_T
    global _OFFENSE_PHASE, _OFFENSE_PHASE_START_T, _OFFENSE_RAM_START_EDGE
    global _OFFENSE_ATTEMPT, _OFFENSE_COOLDOWN_UNTIL, _POLICY_DEBUG
    _LAST_T = None
    _LAST_ACTIONS = dict(_ZERO_WHEEL_MAP)
    _LAST_LINEAR_CMD = 0.0
    _LAST_YAW_CMD = 0.0
    _DEADLOCK_COUNT = 0
    _BOOST_UNTIL = 0
    _SLIP_INT = 0.0
    _WHEEL_INT = dict(_ZERO_WHEEL_MAP)
    _WHEEL_LAST_ERR = dict(_ZERO_WHEEL_MAP)
    _WHEEL_LAST_T = None
    _EDGE_RECOVERY_LATCH = False
    _ORBIT_DIRECTION = 0
    _ORBIT_DIRECTION_LOCK_T = -10**9
    _OFFENSE_PHASE = "idle"
    _OFFENSE_PHASE_START_T = -10**9
    _OFFENSE_RAM_START_EDGE = None
    _OFFENSE_ATTEMPT = 0
    _OFFENSE_COOLDOWN_UNTIL = -10**9
    _POLICY_DEBUG = {}


def _clip(x, lo=-1.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _slew_limit(value, prev, max_delta):
    if value > prev + max_delta:
        return prev + max_delta
    if value < prev - max_delta:
        return prev - max_delta
    return value


def _unit_xy(x, y, fallback=(1.0, 0.0)):
    mag = math.hypot(x, y)
    if mag < 1e-6:
        return fallback
    return (x / mag, y / mag)


def _polar_xy(radius, angle):
    return (radius * math.cos(angle), radius * math.sin(angle))


def _clamp_xy_radius(xy, max_radius, min_radius=0.0):
    x = float(xy[0])
    y = float(xy[1])
    radius = math.hypot(x, y)
    if radius < 1e-6:
        return (float(min_radius), 0.0)
    if radius > max_radius:
        scale = max_radius / radius
        x *= scale
        y *= scale
        radius = max_radius
    if radius < min_radius:
        scale = min_radius / radius
        x *= scale
        y *= scale
    return (x, y)


def _set_policy_debug(payload):
    global _POLICY_DEBUG
    _POLICY_DEBUG = dict(payload)


def policy_debug():
    return dict(_POLICY_DEBUG)


def _estimate_ring_radius(my_pos, my_edge, opp_pos, opp_edge):
    candidates = []
    my_radius = math.hypot(float(my_pos[0]), float(my_pos[1]))
    opp_radius = math.hypot(float(opp_pos[0]), float(opp_pos[1]))
    if my_edge is not None:
        candidates.append(my_radius + float(my_edge))
    if opp_edge is not None:
        candidates.append(opp_radius + float(opp_edge))
    if candidates:
        return max(2.5, max(candidates))
    return max(5.0, my_radius, opp_radius)


def _opp_speed_xy(vel):
    return math.hypot(float(vel[0]), float(vel[1]))


def _planar_speed(vel):
    return math.hypot(float(vel[0]), float(vel[1]))


def _opponent_motion_state(obs):
    history = list(obs.get("obs_history", []) or [])
    points = []
    speeds = []
    for entry in history[-18:]:
        pos = entry.get("opponent_pos", None)
        if pos is not None and len(pos) >= 2:
            points.append((float(pos[0]), float(pos[1])))
        vel = entry.get("opponent_velocity", None)
        if vel is not None and len(vel) >= 2:
            speeds.append(_opp_speed_xy(vel))

    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    opp_vel = obs.get("opponent_velocity", [0.0, 0.0, 0.0])
    points.append((float(opp_pos[0]), float(opp_pos[1])))
    speeds.append(_opp_speed_xy(opp_vel))

    if len(points) < 8:
        return {
            "window_steps": max(0, len(points) - 1),
            "path_length": 0.0,
            "net_displacement": 0.0,
            "avg_speed": speeds[-1] if speeds else 0.0,
            "max_speed": max(speeds) if speeds else 0.0,
            "offense_ready": False,
            "stalled": False,
        }

    path_length = 0.0
    for idx in range(1, len(points)):
        dx = points[idx][0] - points[idx - 1][0]
        dy = points[idx][1] - points[idx - 1][1]
        path_length += math.hypot(dx, dy)
    net_displacement = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    avg_speed = sum(speeds) / max(1, len(speeds))
    max_speed = max(speeds) if speeds else 0.0

    stalled = (
        path_length < 0.65
        and net_displacement < 0.34
        and avg_speed < 0.18
        and max_speed < 0.36
    )
    barely_moved = (
        path_length < 1.20
        and net_displacement < 0.60
        and avg_speed < 0.30
        and max_speed < 0.60
    )
    return {
        "window_steps": len(points) - 1,
        "path_length": path_length,
        "net_displacement": net_displacement,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "offense_ready": stalled or barely_moved,
        "stalled": stalled,
    }


def _self_motion_state(obs):
    history = list(obs.get("obs_history", []) or [])
    points = []
    speeds = []
    opp_edges = []
    opp_dists = []
    for entry in history[-14:]:
        pos = entry.get("my_pos", None)
        if pos is not None and len(pos) >= 2:
            points.append((float(pos[0]), float(pos[1])))
        vel = entry.get("my_velocity", None)
        if vel is not None and len(vel) >= 2:
            speeds.append(_planar_speed(vel))
        opp_edge = entry.get("opponent_edge_distance", None)
        if opp_edge is not None:
            opp_edges.append(float(opp_edge))
        opp_dist = entry.get("distance_to_opponent", None)
        if opp_dist is not None:
            opp_dists.append(float(opp_dist))

    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    my_vel = obs.get("my_velocity", [0.0, 0.0, 0.0])
    opp_edge = obs.get("opponent_edge_distance", None)
    opp_dist = obs.get("distance_to_opponent", None)
    points.append((float(my_pos[0]), float(my_pos[1])))
    speeds.append(_planar_speed(my_vel))
    if opp_edge is not None:
        opp_edges.append(float(opp_edge))
    if opp_dist is not None:
        opp_dists.append(float(opp_dist))

    path_length = 0.0
    for idx in range(1, len(points)):
        dx = points[idx][0] - points[idx - 1][0]
        dy = points[idx][1] - points[idx - 1][1]
        path_length += math.hypot(dx, dy)
    avg_speed = sum(speeds) / max(1, len(speeds))
    max_speed = max(speeds) if speeds else 0.0
    opp_edge_progress = 0.0
    if len(opp_edges) >= 2:
        opp_edge_progress = opp_edges[0] - opp_edges[-1]
    opp_dist_progress = 0.0
    if len(opp_dists) >= 2:
        opp_dist_progress = opp_dists[0] - opp_dists[-1]
    return {
        "window_steps": max(0, len(points) - 1),
        "path_length": path_length,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "opp_edge_progress": opp_edge_progress,
        "opp_dist_progress": opp_dist_progress,
    }


def _set_offense_phase(phase, t, opp_edge=None, *, cooldown_steps=0):
    global _OFFENSE_PHASE, _OFFENSE_PHASE_START_T, _OFFENSE_RAM_START_EDGE
    global _OFFENSE_ATTEMPT, _OFFENSE_COOLDOWN_UNTIL

    prev_phase = _OFFENSE_PHASE
    _OFFENSE_PHASE = phase
    _OFFENSE_PHASE_START_T = t
    if cooldown_steps > 0:
        _OFFENSE_COOLDOWN_UNTIL = max(_OFFENSE_COOLDOWN_UNTIL, t + int(cooldown_steps))

    if phase == "idle":
        _OFFENSE_RAM_START_EDGE = None
        _OFFENSE_ATTEMPT = 0
        return

    if phase == "stage":
        if prev_phase in {"idle", ""}:
            _OFFENSE_ATTEMPT = 1
        elif prev_phase == "backoff":
            _OFFENSE_ATTEMPT += 1
        _OFFENSE_RAM_START_EDGE = None
        return

    if phase == "ram":
        _OFFENSE_RAM_START_EDGE = None if opp_edge is None else float(opp_edge)


def _offense_geometry(obs, ring_radius, safe_radius):
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    opp_edge = obs.get("opponent_edge_distance", None)
    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])
    fallback = _unit_xy(opp_x - my_x, opp_y - my_y, fallback=(1.0, 0.0))
    outward_x, outward_y = _unit_xy(opp_x, opp_y, fallback=fallback)
    inward_x, inward_y = (-outward_x, -outward_y)
    radial_angle = math.atan2(outward_y, outward_x)

    stage_gap = 0.96
    backoff_gap = 1.40
    edge_close = 0.0 if opp_edge is None else _clip((1.15 - float(opp_edge)) / 1.15, 0.0, 1.0)
    ram_approach_gap = 0.20 + 0.16 * (1.0 - edge_close)
    ram_contact_gap = 0.24 + 0.14 * edge_close
    ram_follow_gap = 0.74 + 0.42 * edge_close
    min_radius = 0.70
    stage_xy = _clamp_xy_radius(
        (opp_x + inward_x * stage_gap, opp_y + inward_y * stage_gap),
        safe_radius,
        min_radius=min_radius,
    )
    backoff_xy = _clamp_xy_radius(
        (opp_x + inward_x * backoff_gap, opp_y + inward_y * backoff_gap),
        safe_radius,
        min_radius=min_radius,
    )
    ram_approach_xy = _clamp_xy_radius(
        (opp_x + inward_x * ram_approach_gap, opp_y + inward_y * ram_approach_gap),
        safe_radius,
        min_radius=min_radius,
    )
    ram_contact_xy = _clamp_xy_radius(
        (opp_x + outward_x * ram_contact_gap, opp_y + outward_y * ram_contact_gap),
        ring_radius + 0.12,
        min_radius=min_radius,
    )
    ram_follow_xy = _clamp_xy_radius(
        (opp_x + outward_x * ram_follow_gap, opp_y + outward_y * ram_follow_gap),
        ring_radius + 0.34,
        min_radius=min_radius,
    )
    return {
        "outward": (outward_x, outward_y),
        "inward": (inward_x, inward_y),
        "radial_angle": radial_angle,
        "stage_xy": stage_xy,
        "backoff_xy": backoff_xy,
        "ram_approach_xy": ram_approach_xy,
        "ram_contact_xy": ram_contact_xy,
        "ram_follow_xy": ram_follow_xy,
    }


def _offense_phase_plan(obs, ring_radius, safe_radius, edge_recover, motion_state):
    global _OFFENSE_PHASE, _OFFENSE_PHASE_START_T, _OFFENSE_RAM_START_EDGE
    global _OFFENSE_ATTEMPT, _OFFENSE_COOLDOWN_UNTIL

    t = int(obs.get("t", 0))
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    my_edge = obs.get("my_edge_distance", None)
    opp_edge = obs.get("opponent_edge_distance", None)
    opp_contact = bool(obs.get("opponent_contact", False))

    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])
    dist_opp = math.hypot(opp_x - my_x, opp_y - my_y)
    my_radius = math.hypot(my_x, my_y)
    opp_radius = math.hypot(opp_x, opp_y)
    late_match_force = t >= 1000
    offense_ready = motion_state["offense_ready"] or late_match_force
    offense_reason = "late-match" if late_match_force and not motion_state["offense_ready"] else "stalled"
    # A stalled opponent is safe to attack from anywhere in the ring; waiting for
    # proximity keeps the car orbiting near the edge and can flip it first.
    engagement_ready = late_match_force or motion_state["offense_ready"] or dist_opp < 7.0
    force_ram_engaged = late_match_force or _OFFENSE_PHASE in {"ram", "backoff"}

    if (edge_recover or (my_edge is not None and float(my_edge) < 0.90)) and not force_ram_engaged:
        if _OFFENSE_PHASE != "idle":
            _set_offense_phase("idle", t, opp_edge, cooldown_steps=45)
        return None

    if opp_edge is not None and float(opp_edge) < -0.02:
        if _OFFENSE_PHASE != "idle":
            _set_offense_phase("idle", t, opp_edge)
        return None

    if _OFFENSE_PHASE == "idle":
        if (
            t >= max(_OFFENSE_COOLDOWN_UNTIL, 60)
            and bool(obs.get("ground_contact", True))
            and offense_ready
            and engagement_ready
        ):
            _set_offense_phase("ram" if late_match_force else "stage", t, opp_edge)
        else:
            return None

    if _OFFENSE_ATTEMPT >= 4 and not offense_ready and not opp_contact:
        _set_offense_phase("idle", t, opp_edge, cooldown_steps=70)
        return None

    geometry = _offense_geometry(obs, ring_radius, safe_radius)
    stage_metrics = _body_frame_metrics(obs, geometry["stage_xy"])
    backoff_metrics = _body_frame_metrics(obs, geometry["backoff_xy"])
    ram_metrics = _body_frame_metrics(obs, geometry["ram_contact_xy"])
    self_motion = _self_motion_state(obs)
    my_angle = math.atan2(my_y, my_x) if my_radius > 1e-6 else geometry["radial_angle"]
    opp_angle = geometry["radial_angle"]
    radial_alignment = abs(_wrap_pi(my_angle - opp_angle))
    aligned_behind = my_radius + 0.15 < opp_radius and radial_alignment < 0.26 and dist_opp < 1.55
    ram_ready = (
        aligned_behind
        or (radial_alignment < 0.46 and dist_opp < 2.05)
        or (abs(stage_metrics["bearing"]) < 0.62 and dist_opp < 2.35)
        or stage_metrics["dist"] < 0.55
    )

    phase = _OFFENSE_PHASE
    phase_elapsed = t - _OFFENSE_PHASE_START_T
    if phase == "stage":
        if ram_ready or phase_elapsed > 60:
            _set_offense_phase("ram", t, opp_edge)
            phase = _OFFENSE_PHASE
            phase_elapsed = 0
    elif phase == "ram":
        progress = 0.0
        if _OFFENSE_RAM_START_EDGE is not None and opp_edge is not None:
            progress = max(0.0, float(_OFFENSE_RAM_START_EDGE) - float(opp_edge))
        ram_stuck = (
            phase_elapsed > 24
            and (opp_contact or dist_opp < 1.25)
            and self_motion["path_length"] < 0.18
            and self_motion["avg_speed"] < 0.10
            and self_motion["max_speed"] < 0.18
            and self_motion["opp_edge_progress"] < 0.05
            and self_motion["opp_dist_progress"] < 0.08
        )
        if late_match_force:
            should_backoff = ram_stuck
        else:
            chase_timeout = 26
            progress_timeout = 56
            phase_timeout = 92
            distance_limit = 1.15
            progress_limit = 0.18
            should_backoff = ram_stuck or (
                (phase_elapsed > chase_timeout and not opp_contact and dist_opp > distance_limit)
                or (phase_elapsed > progress_timeout and progress < progress_limit)
                or phase_elapsed > phase_timeout
            )
        if should_backoff:
            _set_offense_phase("backoff", t, opp_edge)
            phase = _OFFENSE_PHASE
            phase_elapsed = 0
    elif phase == "backoff":
        backoff_done = backoff_metrics["dist"] < 0.45 or phase_elapsed > (28 if late_match_force else 62)
        if backoff_done:
            if _OFFENSE_ATTEMPT >= 4 and not offense_ready:
                _set_offense_phase("idle", t, opp_edge, cooldown_steps=70)
                return None
            _set_offense_phase("ram" if late_match_force else "stage", t, opp_edge)
            phase = _OFFENSE_PHASE

    if _OFFENSE_PHASE == "idle":
        return None

    if phase == "stage":
        mode = "offense-stage"
        waypoints = [geometry["stage_xy"], geometry["ram_approach_xy"], geometry["ram_contact_xy"]]
        aim_target_xy = _aim_target_from_waypoints(obs, waypoints, 0.42)
        boost_active = False
    elif phase == "ram":
        mode = "offense-ram"
        waypoints = [geometry["ram_approach_xy"], geometry["ram_contact_xy"], geometry["ram_follow_xy"], geometry["backoff_xy"]]
        if not opp_contact and dist_opp > 2.2:
            aim_target_xy = geometry["ram_approach_xy"]
        elif opp_contact or dist_opp < 0.2:
            aim_target_xy = geometry["ram_follow_xy"]
        else:
            aim_target_xy = geometry["ram_contact_xy"]
        boost_active = True
    else:
        mode = "offense-backoff"
        waypoints = [geometry["backoff_xy"], geometry["stage_xy"], geometry["ram_approach_xy"]]
        aim_target_xy = _aim_target_from_waypoints(obs, waypoints, 0.42)
        boost_active = False

    capture_radius = 0.42 if phase != "ram" else 0.56
    return {
        "mode": mode,
        "waypoints": [tuple(point) for point in waypoints],
        "aim_target_xy": aim_target_xy,
        "capture_radius": capture_radius,
        "orbit_radius": ring_radius - 0.95,
        "ring_radius": ring_radius,
        "opponent_distance": dist_opp,
        "orbit_direction": "n/a",
        "offense_attempt": _OFFENSE_ATTEMPT,
        "stall_path_length": motion_state["path_length"],
        "offense_reason": offense_reason,
        "late_match_force": late_match_force,
        "boost_active": boost_active,
        "ram_stuck": phase == "ram" and late_match_force and self_motion["path_length"] < 0.18,
    }


def _orbit_waypoints(base_theta, current_radius, orbit_radius, direction, *, count=3):
    steps = (0.34, 0.82, 1.28, 1.70)
    radial_error = orbit_radius - current_radius
    entry_radius = current_radius + _clip(radial_error, -0.85, 0.35)
    entry_radius = _clip(entry_radius, 1.0, orbit_radius)
    waypoints = []
    for idx, step in enumerate(steps[:max(1, int(count))]):
        angle = base_theta + direction * step
        radius = entry_radius if idx == 0 else orbit_radius
        waypoints.append(_polar_xy(radius, angle))
    return waypoints


def _escape_waypoint(my_xy, opp_xy, my_theta, orbit_radius, safe_radius, direction, panic):
    away_x, away_y = _unit_xy(
        float(my_xy[0]) - float(opp_xy[0]),
        float(my_xy[1]) - float(opp_xy[1]),
        fallback=_polar_xy(1.0, my_theta + math.pi),
    )
    inward_x, inward_y = _unit_xy(-float(my_xy[0]), -float(my_xy[1]), fallback=(-1.0, 0.0))
    tan_x, tan_y = _unit_xy(-math.sin(my_theta) * direction, math.cos(my_theta) * direction)

    away_w = 1.05 if panic else 0.88
    inward_w = 1.25 if panic else 0.92
    tan_w = 0.42 if panic else 0.62
    dir_x = away_w * away_x + inward_w * inward_x + tan_w * tan_x
    dir_y = away_w * away_y + inward_w * inward_y + tan_w * tan_y
    dir_x, dir_y = _unit_xy(dir_x, dir_y, fallback=(inward_x, inward_y))

    step = 1.35 if panic else 1.00
    candidate = (
        float(my_xy[0]) + dir_x * step,
        float(my_xy[1]) + dir_y * step,
    )
    min_radius = max(1.15, 0.55 * orbit_radius)
    candidate = _clamp_xy_radius(candidate, safe_radius, min_radius=min_radius)

    cand_theta = math.atan2(candidate[1], candidate[0]) if math.hypot(candidate[0], candidate[1]) > 1e-6 else my_theta
    orbit_bias_radius = min(safe_radius, max(min_radius, orbit_radius - (0.30 if panic else 0.12)))
    orbit_bias = _polar_xy(orbit_bias_radius, cand_theta + direction * (0.10 if panic else 0.06))
    blended = (
        candidate[0] * 0.78 + orbit_bias[0] * 0.22,
        candidate[1] * 0.78 + orbit_bias[1] * 0.22,
    )
    return _clamp_xy_radius(blended, safe_radius, min_radius=min_radius)


def _recovery_waypoint(my_theta, recover_radius, direction):
    return _polar_xy(recover_radius, my_theta + direction * 0.18)


def _turn_cost_to_waypoint(obs, waypoint_xy):
    metrics = _body_frame_metrics(obs, waypoint_xy)
    reverse_bearing = _wrap_pi(metrics["bearing"] - math.pi)
    return min(abs(metrics["bearing"]), abs(reverse_bearing))


def _score_waypoint_plan(obs, opp_pos, my_theta, direction, orbit_radius, waypoints, prefer_escape=False):
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])
    opp_theta = math.atan2(opp_y, opp_x)
    current_sep = abs(_wrap_pi(opp_theta - my_theta))
    weights = (1.0, 0.55, 0.30)
    score = 0.0
    for idx, waypoint in enumerate(waypoints[:3]):
        weight = weights[idx]
        score += weight * math.hypot(waypoint[0] - opp_x, waypoint[1] - opp_y)
        if idx == 0:
            score -= 0.34 * _turn_cost_to_waypoint(obs, waypoint)
            future_theta = math.atan2(waypoint[1], waypoint[0]) if math.hypot(waypoint[0], waypoint[1]) > 1e-6 else my_theta
            score += (1.25 if prefer_escape else 0.35) * (
                abs(_wrap_pi(opp_theta - future_theta)) - current_sep
            )
            score -= 0.12 * abs(math.hypot(waypoint[0], waypoint[1]) - orbit_radius)
    if direction == _ORBIT_DIRECTION:
        score += 0.18
    return score


def _build_direction_plan(obs, direction, orbit_radius, safe_radius, plan_mode, panic):
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    my_theta = math.atan2(my_y, my_x) if math.hypot(my_x, my_y) > 1e-6 else float(obs.get("my_yaw", 0.0))
    my_radius = math.hypot(my_x, my_y)

    if plan_mode == "edge-recover":
        recover_radius = max(1.0, min(safe_radius, orbit_radius - 0.55))
        first = _recovery_waypoint(my_theta, recover_radius, direction)
        base_theta = math.atan2(first[1], first[0]) if math.hypot(first[0], first[1]) > 1e-6 else my_theta
        trail = _orbit_waypoints(base_theta, recover_radius, max(recover_radius, orbit_radius - 0.15), direction, count=2)
        waypoints = [first] + trail
    elif plan_mode == "evade":
        first = _escape_waypoint(
            (my_x, my_y),
            (float(opp_pos[0]), float(opp_pos[1])),
            my_theta,
            orbit_radius,
            safe_radius,
            direction,
            panic,
        )
        first_radius = math.hypot(first[0], first[1])
        base_theta = math.atan2(first[1], first[0]) if first_radius > 1e-6 else my_theta
        trail = _orbit_waypoints(base_theta, first_radius, orbit_radius, direction, count=2)
        waypoints = [first] + trail
    else:
        waypoints = _orbit_waypoints(my_theta, my_radius, orbit_radius, direction, count=3)

    score = _score_waypoint_plan(
        obs,
        opp_pos,
        my_theta,
        direction,
        orbit_radius,
        waypoints,
        prefer_escape=(plan_mode != "orbit"),
    )
    return {
        "direction": direction,
        "waypoints": waypoints,
        "score": score,
    }


def _choose_orbit_direction(obs, orbit_radius, safe_radius, plan_mode, panic):
    global _ORBIT_DIRECTION, _ORBIT_DIRECTION_LOCK_T

    t = int(obs.get("t", 0))
    candidates = {
        direction: _build_direction_plan(obs, direction, orbit_radius, safe_radius, plan_mode, panic)
        for direction in (1, -1)
    }
    best_direction = max(candidates, key=lambda direction: candidates[direction]["score"])
    if _ORBIT_DIRECTION == 0:
        _ORBIT_DIRECTION = best_direction
        _ORBIT_DIRECTION_LOCK_T = t + 24
    elif best_direction != _ORBIT_DIRECTION:
        current_score = candidates[_ORBIT_DIRECTION]["score"]
        best_score = candidates[best_direction]["score"]
        score_margin = 0.36 if plan_mode != "orbit" else 0.62
        if t >= _ORBIT_DIRECTION_LOCK_T and best_score > current_score + score_margin:
            _ORBIT_DIRECTION = best_direction
            _ORBIT_DIRECTION_LOCK_T = t + (30 if plan_mode != "orbit" else 44)
    return candidates[_ORBIT_DIRECTION]


def _aim_target_from_waypoints(obs, waypoints, capture_radius):
    if not waypoints:
        return (0.0, 0.0)
    first = waypoints[0]
    if len(waypoints) == 1:
        return first

    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    first_distance = math.hypot(first[0] - float(my_pos[0]), first[1] - float(my_pos[1]))
    blend_start = max(capture_radius + 0.70, 1.10)
    if first_distance >= blend_start:
        return first

    alpha = _clip(
        (blend_start - first_distance) / max(1e-6, blend_start - capture_radius),
        0.0,
        0.60,
    )
    second = waypoints[1]
    return (
        first[0] * (1.0 - alpha) + second[0] * alpha,
        first[1] * (1.0 - alpha) + second[1] * alpha,
    )


def _sumo_waypoint_profile(plan_mode, first_distance):
    if plan_mode == "offense-stage":
        return _profile_with_updates(
            _sequence_profile(0.42, precise=first_distance < 0.85),
            max_speed=2.55,
            min_speed=0.35,
            stop_radius=0.06,
            keep_moving_angle=1.40,
            max_yaw_rate=12.8,
            turn_gain=1.38,
            turn_linear_drop=0.14,
            turn_linear_floor=0.12,
            turn_diff_boost=0.32,
            inside_feather_gain=0.18,
            max_cmd=1.0,
            max_step=0.22,
            linear_slew_up=0.24,
            linear_slew_down=0.42,
            yaw_slew=0.54,
        )
    if plan_mode == "offense-ram":
        return _profile_with_updates(
            _sequence_profile(0.60),
            max_speed=4.0,
            min_speed=2.15,
            stop_radius=0.03,
            pivot_angle=1.72,
            pivot_distance=0.78,
            keep_moving_angle=1.86,
            allow_reverse=False,
            lookahead_min=0.86,
            lookahead_max=2.10,
            lookahead_speed_gain=0.22,
            max_curvature=4.0,
            max_lateral_acc=7.6,
            max_yaw_rate=11.2,
            min_align_scale=0.68,
            align_heading_gain=0.60,
            turn_gain=1.18,
            turn_linear_drop=0.0,
            turn_linear_floor=0.62,
            turn_diff_boost=0.10,
            inside_feather_gain=0.02,
            max_cmd=1.0,
            max_step=0.30,
            linear_slew_up=0.52,
            linear_slew_down=0.42,
            yaw_slew=0.26,
            turn_slow_start=0.72,
            turn_slow_end=1.42,
            turn_slow_floor=0.55,
            yaw_error_slow_start=0.54,
            yaw_error_floor=0.46,
            yaw_error_gain=0.34,
            boost_speed_scale=1.08,
            boost_max_cmd=1.0,
            boost_max_step=0.34,
            edge_brake_acc=12.5,
            edge_base_buffer=0.12,
            edge_hard_buffer=0.22,
            edge_caution_buffer=0.68,
            edge_reaction_time=0.08,
            edge_turn_time_gain=0.46,
            edge_rescue_speed=1.5,
            edge_attack_enable=True,
            edge_attack_dist=2.35,
            edge_attack_bearing=1.05,
            edge_attack_forward_cos=0.16,
            edge_attack_opp_edge=1.35,
            edge_attack_self_margin=0.64,
            edge_attack_self_span=1.00,
            edge_attack_extra_outward_speed=2.2,
            edge_attack_risk_relief=0.88,
            edge_attack_yaw_relief=0.92,
            edge_attack_rescue_margin=0.52,
        )
    if plan_mode == "offense-backoff":
        return _profile_with_updates(
            _sequence_profile(0.46, cautious=True),
            max_speed=2.25,
            min_speed=0.15,
            stop_radius=0.06,
            keep_moving_angle=1.36,
            max_yaw_rate=11.0,
            turn_gain=1.30,
            turn_linear_drop=0.18,
            turn_linear_floor=0.10,
            turn_diff_boost=0.26,
            inside_feather_gain=0.24,
            max_cmd=1.0,
            max_step=0.22,
            linear_slew_up=0.22,
            linear_slew_down=0.48,
            yaw_slew=0.48,
        )
    if plan_mode == "edge-recover":
        return _profile_with_updates(
            _target_chase_profile(0.45, rescue=True),
            max_speed=1.9,
            stop_radius=0.08,
            keep_moving_angle=1.18,
            turn_gain=1.30,
            turn_diff_boost=0.34,
            inside_feather_gain=0.26,
        )
    if plan_mode == "evade":
        return _profile_with_updates(
            _sequence_profile(0.55, cautious=first_distance < 1.20),
            max_speed=2.55,
            min_speed=0.45,
            stop_radius=0.10,
            keep_moving_angle=1.34,
            max_yaw_rate=11.5,
            turn_gain=1.24,
            turn_linear_drop=0.22,
            turn_linear_floor=0.10,
            turn_diff_boost=0.30,
            inside_feather_gain=0.26,
            max_cmd=1.0,
            max_step=0.22,
            linear_slew_up=0.24,
            linear_slew_down=0.46,
            yaw_slew=0.46,
        )
    precise = first_distance < 0.95
    return _profile_with_updates(
        _sequence_profile(0.55, precise=precise),
        max_speed=3.0,
        min_speed=1.10,
        stop_radius=0.05,
        keep_moving_angle=1.35,
        lookahead_min=0.80,
        lookahead_max=2.30,
        lookahead_speed_gain=0.26,
        max_curvature=4.2,
        max_lateral_acc=7.0,
        max_yaw_rate=11.0,
        turn_gain=1.14,
        turn_linear_drop=0.12,
        turn_linear_floor=0.24,
        turn_diff_boost=0.14,
        inside_feather_gain=0.16,
        max_cmd=1.0,
        max_step=0.22,
        linear_slew_up=0.28,
        linear_slew_down=0.44,
        yaw_slew=0.40,
    )


def _sumo_waypoint_plan(obs):
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    my_edge = obs.get("my_edge_distance", None)
    opp_edge = obs.get("opponent_edge_distance", None)
    opp_contact = bool(obs.get("opponent_contact", False))
    metrics = _body_frame_metrics(obs, (float(opp_pos[0]), float(opp_pos[1])))
    edge_state = _edge_state(obs, allow_reverse=True)

    ring_radius = _estimate_ring_radius(my_pos, my_edge, opp_pos, opp_edge)
    speed_xy = metrics["speed_xy"]
    orbit_margin = 0.92 + 0.10 * min(speed_xy, 2.0)
    if my_edge is not None:
        orbit_margin += 0.18 * _clip(1.0 - float(my_edge), 0.0, 1.2)
    orbit_radius = max(1.45, ring_radius - orbit_margin)
    safe_radius = max(1.25, ring_radius - 1.25)

    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])
    dist_opp = math.hypot(opp_x - my_x, opp_y - my_y)
    my_theta = math.atan2(my_y, my_x) if math.hypot(my_x, my_y) > 1e-6 else float(obs.get("my_yaw", 0.0))
    opp_theta = math.atan2(opp_y, opp_x) if math.hypot(opp_x, opp_y) > 1e-6 else 0.0
    angular_sep = abs(_wrap_pi(opp_theta - my_theta))
    motion_state = _opponent_motion_state(obs)

    edge_recover = False
    if my_edge is not None:
        edge_recover = float(my_edge) < 0.90
    if edge_state is not None:
        edge_recover = edge_recover or (
            edge_state["edge_distance"] < 1.20 and edge_state["radial_speed_out"] > 0.18
        )

    offense_plan = _offense_phase_plan(obs, ring_radius, safe_radius, edge_recover, motion_state)
    if offense_plan is not None:
        return offense_plan

    evade = dist_opp < 2.35 or (dist_opp < 2.85 and angular_sep < 0.60) or opp_contact
    panic = dist_opp < 1.55 or opp_contact
    if edge_recover:
        plan_mode = "edge-recover"
    elif evade:
        plan_mode = "evade"
    else:
        plan_mode = "orbit"

    plan = _choose_orbit_direction(obs, orbit_radius, safe_radius, plan_mode, panic)
    waypoints = [tuple(waypoint) for waypoint in plan["waypoints"]]
    capture_radius = 0.58 if plan_mode == "orbit" else 0.52
    aim_target_xy = _aim_target_from_waypoints(obs, waypoints, capture_radius)

    return {
        "mode": plan_mode,
        "waypoints": waypoints,
        "aim_target_xy": aim_target_xy,
        "capture_radius": capture_radius,
        "orbit_radius": orbit_radius,
        "ring_radius": ring_radius,
        "opponent_distance": dist_opp,
        "orbit_direction": "ccw" if plan["direction"] > 0 else "cw",
        "offense_attempt": 0,
        "stall_path_length": motion_state["path_length"],
        "offense_reason": "",
        "late_match_force": False,
    }


def _circle_target(my_pos, opp_pos, my_edge):
    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])

    vx = my_x - opp_x
    vy = my_y - opp_y
    dist = math.hypot(vx, vy)
    if dist < 1e-6:
        vx, vy, dist = 1.0, 0.0, 1.0
    nx = vx / dist
    ny = vy / dist

    tx = -ny
    ty = nx

    opp_r = math.hypot(opp_x, opp_y)
    if opp_r > 1e-6:
        cx = -opp_x / opp_r
        cy = -opp_y / opp_r
    else:
        cx, cy = 1.0, 0.0

    bias = 0.35
    dir1x = tx + cx * bias
    dir1y = ty + cy * bias
    dir2x = -tx + cx * bias
    dir2y = -ty + cy * bias
    d1 = math.hypot(dir1x, dir1y)
    d2 = math.hypot(dir2x, dir2y)
    if d1 < 1e-6:
        dir1x, dir1y, d1 = 1.0, 0.0, 1.0
    if d2 < 1e-6:
        dir2x, dir2y, d2 = -1.0, 0.0, 1.0
    dir1x /= d1
    dir1y /= d1
    dir2x /= d2
    dir2y /= d2

    circle_radius = min(max(dist, 1.4), 2.2)
    t1x = opp_x + dir1x * circle_radius
    t1y = opp_y + dir1y * circle_radius
    t2x = opp_x + dir2x * circle_radius
    t2y = opp_y + dir2y * circle_radius

    if math.hypot(t1x, t1y) <= math.hypot(t2x, t2y):
        tx_sel, ty_sel = t1x, t1y
    else:
        tx_sel, ty_sel = t2x, t2y

    if my_edge is not None:
        my_r = math.hypot(my_x, my_y)
        ring_r = my_r + float(my_edge)
        max_r = max(0.5, ring_r - 0.8)
        sel_r = math.hypot(tx_sel, ty_sel)
        if sel_r > max_r and sel_r > 1e-6:
            scale = max_r / sel_r
            tx_sel *= scale
            ty_sel *= scale

    return (tx_sel, ty_sel)


def _select_sumo_target(my_pos, opp_pos, my_edge, opp_edge):
    my_x = float(my_pos[0])
    my_y = float(my_pos[1])
    opp_x = float(opp_pos[0])
    opp_y = float(opp_pos[1])

    my_r = math.hypot(my_x, my_y)
    opp_r = math.hypot(opp_x, opp_y)
    dist_opp = math.hypot(opp_x - my_x, opp_y - my_y)
    if opp_r > 1e-6:
        opp_rad_x = opp_x / opp_r
        opp_rad_y = opp_y / opp_r
    else:
        opp_rad_x = 1.0
        opp_rad_y = 0.0

    safe_margin = 0.8
    circle_margin = 0.8
    inside_margin = 0.6
    reposition_offset = 0.9
    push_offset = 0.25

    if my_edge is not None and float(my_edge) < safe_margin:
        return (0.0, 0.0), "recover"

    inside_adv = opp_r - my_r
    if inside_adv < inside_margin and (my_edge is None or float(my_edge) > circle_margin):
        if 0.6 < dist_opp < 3.5:
            return _circle_target(my_pos, opp_pos, my_edge), "circle"
        offset = min(reposition_offset, max(0.0, opp_r - 0.2))
        return (opp_x - opp_rad_x * offset, opp_y - opp_rad_y * offset), "reposition"

    if my_edge is None or float(my_edge) > circle_margin:
        return _circle_target(my_pos, opp_pos, my_edge), "circle"

    offset = min(push_offset, max(0.0, opp_r - 0.1))
    return (opp_x - opp_rad_x * offset, opp_y - opp_rad_y * offset), "push"


def _body_frame_metrics(obs, target_xy):
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    my_yaw = float(obs.get("my_yaw", 0.0))
    vel = obs.get("my_velocity", [0.0, 0.0, 0.0])
    ang_vel = obs.get("my_angular_velocity", [0.0, 0.0, 0.0])

    dx = float(target_xy[0]) - float(my_pos[0])
    dy = float(target_xy[1]) - float(my_pos[1])
    cos_yaw = math.cos(my_yaw)
    sin_yaw = math.sin(my_yaw)
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    dist = math.hypot(dx, dy)
    bearing = math.atan2(body_y, body_x)
    forward_speed = float(vel[0]) * cos_yaw + float(vel[1]) * sin_yaw
    lateral_speed = -float(vel[0]) * sin_yaw + float(vel[1]) * cos_yaw
    speed_xy = math.hypot(float(vel[0]), float(vel[1]))
    yaw_rate = float(ang_vel[2]) if len(ang_vel) >= 3 else 0.0
    return {
        "dx": dx,
        "dy": dy,
        "dist": dist,
        "body_x": body_x,
        "body_y": body_y,
        "bearing": bearing,
        "forward_speed": forward_speed,
        "lateral_speed": lateral_speed,
        "speed_xy": speed_xy,
        "yaw_rate": yaw_rate,
    }


def _wheel_omegas_from_obs(act_vel):
    wheel_omegas = []
    for key in _WHEEL_NAMES:
        val = act_vel.get(key, None)
        if val is not None:
            wheel_omegas.append(float(val))
    return wheel_omegas


def _traction_scale(linear_cmd, forward_speed, wheel_omegas):
    global _SLIP_INT

    if not wheel_omegas or abs(linear_cmd) < 0.05:
        _SLIP_INT *= 0.9
        return 1.0

    omega = sum(wheel_omegas) / max(1.0, float(len(wheel_omegas)))
    wheel_speed = omega * _WHEEL_RADIUS
    if abs(forward_speed) > 0.2 and wheel_speed * forward_speed < 0.0:
        wheel_speed = -wheel_speed

    v_ref = max(0.4, abs(forward_speed), abs(linear_cmd))
    slip = (wheel_speed - forward_speed) / v_ref
    slip_target = 0.10 + 0.12 * math.exp(-abs(forward_speed) / 0.8)
    if linear_cmd < 0.0:
        slip_target = -slip_target

    err = slip - slip_target
    _SLIP_INT = _clip(_SLIP_INT + err * 0.06, -0.5, 0.5)
    adjust = 0.65 * err + 0.18 * _SLIP_INT
    return _clip(1.0 - adjust, 0.45, 1.05)


def _wheel_speed_control(targets, act_vel, t, max_cmd, max_step, turn_ratio=0.0):
    global _WHEEL_INT, _WHEEL_LAST_ERR, _WHEEL_LAST_T, _LAST_ACTIONS

    if _WHEEL_LAST_T is None:
        dt = 1.0
    else:
        dt = max(1.0, float(t - _WHEEL_LAST_T))
    _WHEEL_LAST_T = t

    commands = {}
    for key, target_frac in targets.items():
        target_frac = _clip(target_frac, -1.0, 1.0)
        target_omega = target_frac * _MAX_WHEEL_OMEGA
        actual_omega = float(act_vel.get(key, 0.0))
        err = target_omega - actual_omega
        prev_err = _WHEEL_LAST_ERR.get(key, 0.0)

        if abs(target_frac) < 0.03:
            _WHEEL_INT[key] *= 0.55
        elif target_omega * actual_omega < 0.0:
            _WHEEL_INT[key] *= 0.35

        _WHEEL_INT[key] = _clip(_WHEEL_INT[key] + err * dt, -20.0, 20.0)
        derr = (err - prev_err) / dt
        _WHEEL_LAST_ERR[key] = err

        ff = (0.56 + 0.05 * turn_ratio) * target_frac
        if abs(target_frac) > 0.07 and abs(actual_omega) < 0.5:
            ff += 0.12 if target_frac > 0.0 else -0.12

        cmd = ff + (0.040 + 0.004 * turn_ratio) * err + 0.005 * _WHEEL_INT[key] + 0.0015 * derr
        if abs(target_frac) < 0.04:
            cmd -= 0.03 * actual_omega

        cmd = _clip(cmd, -max_cmd, max_cmd)
        prev_cmd = _LAST_ACTIONS.get(key, 0.0)
        cmd = _slew_limit(cmd, prev_cmd, max_step * (1.0 + 0.10 * turn_ratio))
        commands[key] = float(cmd)
        _LAST_ACTIONS[key] = float(cmd)

    return commands


def _wheel_targets_from_body_command(linear_speed, yaw_rate, profile):
    turn_speed = yaw_rate * _TRACK_HALF * profile.get("turn_gain", 1.0)
    turn_mag = abs(turn_speed)
    linear_mag = abs(linear_speed)
    turn_ratio = turn_mag / max(0.20, linear_mag + turn_mag)

    linear_keep = 1.0 - profile.get("turn_linear_drop", 0.0) * turn_ratio
    linear_keep = max(profile.get("turn_linear_floor", 0.18), min(1.0, linear_keep))
    shaped_linear = linear_speed * linear_keep
    shaped_turn = turn_speed * (1.0 + profile.get("turn_diff_boost", 0.0) * turn_ratio)

    left_speed = shaped_linear - shaped_turn
    right_speed = shaped_linear + shaped_turn

    feather_start = profile.get("inside_feather_start", 0.24)
    if left_speed * right_speed > 0.0 and turn_ratio > feather_start:
        feather_strength = (turn_ratio - feather_start) / max(1e-6, 1.0 - feather_start)
        inside_scale = 1.0 - profile.get("inside_feather_gain", 0.0) * feather_strength
        inside_scale = max(profile.get("inside_feather_floor", 0.12), inside_scale)
        if abs(left_speed) < abs(right_speed):
            left_speed *= inside_scale
        else:
            right_speed *= inside_scale

    max_abs = max(abs(left_speed), abs(right_speed), _MAX_WHEEL_LINEAR)
    left_frac = left_speed / max_abs
    right_frac = right_speed / max_abs
    return {
        "motor_fl": float(_clip(left_frac)),
        "motor_rl": float(_clip(left_frac)),
        "motor_fr": float(_clip(right_frac)),
        "motor_rr": float(_clip(right_frac)),
    }, turn_ratio


def _speed_limit_for_curvature(curvature, max_speed, max_lateral_acc):
    curv = abs(curvature)
    if curv < 1e-4:
        return max_speed
    return min(max_speed, math.sqrt(max(0.0, max_lateral_acc / curv)))


def _speed_limit_for_stop(distance, stop_radius, max_brake_acc):
    if distance <= stop_radius:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * max_brake_acc * (distance - stop_radius)))


def _edge_state(obs, allow_reverse):
    edge_distance = obs.get("my_edge_distance", None)
    if edge_distance is None:
        return None

    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    vel = obs.get("my_velocity", [0.0, 0.0, 0.0])
    yaw = float(obs.get("my_yaw", 0.0))

    x = float(my_pos[0])
    y = float(my_pos[1])
    radius = math.hypot(x, y)
    if radius > 1e-6:
        outward_x = x / radius
        outward_y = y / radius
    else:
        outward_x = 1.0
        outward_y = 0.0

    vx = float(vel[0])
    vy = float(vel[1])
    radial_speed_out = vx * outward_x + vy * outward_y
    tangential_speed = -vx * outward_y + vy * outward_x

    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    forward_outward_cos = forward_x * outward_x + forward_y * outward_y

    inward_heading = math.atan2(-outward_y, -outward_x)
    inward_bearing = _wrap_pi(inward_heading - yaw)
    reverse_inward_bearing = _wrap_pi(inward_bearing - math.pi)
    if allow_reverse and abs(reverse_inward_bearing) + 0.10 < abs(inward_bearing):
        escape_bearing = reverse_inward_bearing
        escape_sign = -1.0
    else:
        escape_bearing = inward_bearing
        escape_sign = 1.0

    return {
        "edge_distance": float(edge_distance),
        "radius": radius,
        "ring_radius": radius + float(edge_distance),
        "radial_speed_out": radial_speed_out,
        "tangential_speed": tangential_speed,
        "forward_outward_cos": forward_outward_cos,
        "inward_bearing": inward_bearing,
        "reverse_inward_bearing": reverse_inward_bearing,
        "escape_bearing": escape_bearing,
        "escape_sign": escape_sign,
    }


def _edge_safety(edge_state, profile):
    if edge_state is None:
        return None

    brake_acc = max(1.0, profile.get("edge_brake_acc", profile.get("max_brake_acc", 8.0)))
    max_yaw_rate = max(1.0, profile.get("edge_turn_rate", profile.get("max_yaw_rate", 5.0)))
    free_bearing = profile.get("edge_free_bearing", 0.18)
    reaction_time = profile.get("edge_reaction_time", 0.08)
    turn_time_gain = profile.get("edge_turn_time_gain", 0.55)
    base_buffer = profile.get("edge_base_buffer", 0.22)
    hard_buffer = profile.get("edge_hard_buffer", base_buffer + 0.10)
    extra_caution = profile.get("edge_extra_caution", 0.18)

    escape_turn_time = max(0.0, abs(edge_state["escape_bearing"]) - free_bearing) / max_yaw_rate
    outward_speed = max(0.0, edge_state["radial_speed_out"])
    turn_distance = outward_speed * (reaction_time + turn_time_gain * escape_turn_time)
    brake_distance = outward_speed * outward_speed / max(2.0 * brake_acc, 1e-6)
    risk_distance = base_buffer + turn_distance + brake_distance

    caution_buffer = max(profile.get("edge_caution_buffer", 0.85), risk_distance + extra_caution)
    rescue_buffer = max(hard_buffer, risk_distance + 0.05)
    caution = edge_state["edge_distance"] <= caution_buffer
    rescue = edge_state["edge_distance"] <= rescue_buffer

    usable_distance = max(0.0, edge_state["edge_distance"] - base_buffer - turn_distance)
    safe_outward_speed = math.sqrt(max(0.0, 2.0 * brake_acc * usable_distance))
    risk = 0.0
    if caution:
        risk = _clip(
            (caution_buffer - edge_state["edge_distance"]) / max(1e-6, caution_buffer - hard_buffer),
            0.0,
            1.0,
        )

    return {
        "caution": caution,
        "rescue": rescue,
        "risk": risk,
        "safe_outward_speed": safe_outward_speed,
        "risk_distance": risk_distance,
        "caution_buffer": caution_buffer,
        "rescue_buffer": rescue_buffer,
    }


def _edge_attack_commit(obs, edge_state, metrics, profile):
    if edge_state is None or not profile.get("edge_attack_enable", False):
        return 0.0

    my_edge = obs.get("my_edge_distance", None)
    opp_edge = obs.get("opponent_edge_distance", None)
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    if my_edge is None or opp_edge is None or len(opp_pos) < 2:
        return 0.0

    my_edge = float(my_edge)
    opp_edge = float(opp_edge)
    self_margin = profile.get("edge_attack_self_margin", 0.70)
    if my_edge <= self_margin:
        return 0.0

    opp_metrics = _body_frame_metrics(obs, (float(opp_pos[0]), float(opp_pos[1])))
    max_dist = profile.get("edge_attack_dist", 2.0)
    if opp_metrics["dist"] > max_dist or abs(opp_metrics["bearing"]) > profile.get("edge_attack_bearing", 1.0):
        return 0.0

    forward_cos = edge_state["forward_outward_cos"]
    min_forward_cos = profile.get("edge_attack_forward_cos", 0.18)
    if forward_cos <= min_forward_cos:
        return 0.0

    opp_edge_trigger = profile.get("edge_attack_opp_edge", 1.0)
    opp_pressure = _clip((opp_edge_trigger - opp_edge) / max(0.25, opp_edge_trigger), 0.0, 1.0)
    if opp_pressure <= 0.0:
        return 0.0

    close_frac = _clip((max_dist - opp_metrics["dist"]) / max(1e-6, max_dist - 0.45), 0.0, 1.0)
    align_frac = _clip((forward_cos - min_forward_cos) / max(1e-6, 1.0 - min_forward_cos), 0.0, 1.0)
    self_frac = _clip((my_edge - self_margin) / profile.get("edge_attack_self_span", 0.85), 0.0, 1.0)
    return min(1.0, opp_pressure * (0.45 + 0.55 * close_frac) * (0.35 + 0.65 * align_frac) * (0.35 + 0.65 * self_frac))


def _limit_outward_linear(desired_linear, forward_outward_cos, safe_outward_speed):
    desired_outward_speed = desired_linear * forward_outward_cos
    if desired_outward_speed <= safe_outward_speed or abs(forward_outward_cos) < 0.05:
        return desired_linear

    limited_mag = safe_outward_speed / max(abs(forward_outward_cos), 1e-6)
    return math.copysign(min(abs(desired_linear), limited_mag), desired_linear)


def _heading_speed_scale(bearing, profile):
    bearing_abs = abs(bearing)
    cos_scale = max(
        profile.get("min_align_scale", 0.18),
        math.cos(min(math.pi / 2, bearing_abs) * profile.get("align_heading_gain", 0.72)),
    )

    slow_start = profile.get("turn_slow_start", 0.45)
    slow_end = max(slow_start + 1e-3, profile.get("turn_slow_end", profile.get("pivot_angle", 1.15)))
    slow_floor = profile.get("turn_slow_floor", profile.get("min_align_scale", 0.18))
    slow_power = profile.get("turn_slow_power", 1.4)
    if bearing_abs <= slow_start:
        return cos_scale

    frac = _clip((bearing_abs - slow_start) / (slow_end - slow_start), 0.0, 1.0)
    turn_scale = 1.0 - (1.0 - slow_floor) * (frac ** slow_power)
    return min(cos_scale, turn_scale)


def _yaw_response_speed_scale(desired_yaw, yaw_rate, profile):
    max_yaw_rate = max(1.0, profile.get("max_yaw_rate", 5.0))
    yaw_error_ratio = abs(desired_yaw - yaw_rate) / max_yaw_rate
    slow_start = profile.get("yaw_error_slow_start", 0.34)
    if yaw_error_ratio <= slow_start:
        return 1.0

    slow_floor = profile.get("yaw_error_floor", 0.22)
    slow_gain = profile.get("yaw_error_gain", 0.72)
    frac = _clip((yaw_error_ratio - slow_start) / max(1e-6, 1.0 - slow_start), 0.0, 1.0)
    return max(slow_floor, 1.0 - slow_gain * frac)


def _compute_body_command(obs, target_xy, profile, boost_active=False):
    global _LAST_T, _LAST_LINEAR_CMD, _LAST_YAW_CMD

    t = int(obs.get("t", 0))
    ground_contact = bool(obs.get("ground_contact", True))
    is_tipping = float(obs.get("is_tipping", 0.0))
    act_vel = obs.get("my_actuator_velocity", {}) or {}
    metrics = _body_frame_metrics(obs, target_xy)
    wheel_omegas = _wheel_omegas_from_obs(act_vel)
    edge_state = _edge_state(obs, allow_reverse=profile.get("allow_reverse", False))
    edge_safety = _edge_safety(edge_state, profile)
    edge_attack_commit = _edge_attack_commit(obs, edge_state, metrics, profile)
    if edge_safety is not None and edge_attack_commit > 0.0:
        edge_safety = dict(edge_safety)
        edge_safety["safe_outward_speed"] += edge_attack_commit * profile.get("edge_attack_extra_outward_speed", 0.0)
        edge_safety["risk"] *= max(0.0, 1.0 - edge_attack_commit * profile.get("edge_attack_risk_relief", 0.7))
        my_edge = obs.get("my_edge_distance", None)
        if my_edge is not None and float(my_edge) > profile.get("edge_attack_rescue_margin", 0.58):
            edge_safety["rescue"] = False
            edge_safety["caution"] = edge_safety["risk"] > 0.02

    distance = metrics["dist"]
    body_x = metrics["body_x"]
    body_y = metrics["body_y"]
    bearing = metrics["bearing"]
    yaw_rate = metrics["yaw_rate"]

    motion_sign = 1.0
    if (
        profile.get("allow_reverse", False)
        and distance <= profile.get("reverse_distance_max", 2.4)
        and abs(bearing) >= profile.get("reverse_angle", 1.85)
    ):
        motion_sign = -1.0
        body_x = -body_x
        body_y = -body_y
        bearing = math.atan2(body_y, body_x)

    lookahead = max(
        profile.get("lookahead_min", 0.6),
        min(
            profile.get("lookahead_max", 1.8),
            distance + profile.get("lookahead_speed_gain", 0.18) * abs(metrics["forward_speed"]),
        ),
    )
    curvature = 2.0 * body_y / max(lookahead * lookahead, 1e-4)
    curvature = _clip(curvature, -profile.get("max_curvature", 3.5), profile.get("max_curvature", 3.5))

    pivot = abs(bearing) > profile.get("pivot_angle", 1.15) and distance > profile.get("pivot_distance", 0.4)
    if body_x < 0.0 and abs(bearing) > profile.get("reverse_forbid_angle", 1.0):
        pivot = True

    max_speed = profile.get("max_speed", 1.6)
    if boost_active:
        max_speed *= profile.get("boost_speed_scale", 1.0)
    stop_speed = _speed_limit_for_stop(distance, profile.get("stop_radius", 0.15), profile.get("max_brake_acc", 8.0))
    curve_speed = _speed_limit_for_curvature(
        curvature,
        max_speed=max_speed,
        max_lateral_acc=profile.get("max_lateral_acc", 5.0),
    )
    align_scale = _heading_speed_scale(bearing, profile)

    if pivot:
        desired_linear = 0.0
    else:
        desired_linear = min(max_speed, stop_speed, curve_speed)
        desired_linear *= align_scale
        if distance > profile.get("stop_radius", 0.15) and abs(bearing) < profile.get("keep_moving_angle", 1.05):
            desired_linear = max(desired_linear, profile.get("min_speed", 0.0))
        desired_linear *= motion_sign

    desired_yaw = profile.get("bearing_k", 4.2) * bearing
    desired_yaw += profile.get("lateral_k", 1.2) * math.atan2(body_y, max(profile.get("lookahead_min", 0.6), abs(body_x) + 0.2))
    desired_yaw -= profile.get("yaw_damp", 0.55) * yaw_rate
    if not pivot:
        desired_yaw += profile.get("curvature_k", 1.0) * desired_linear * curvature
    else:
        desired_yaw = profile.get("pivot_k", 5.4) * math.tanh(1.3 * bearing) - profile.get("yaw_damp", 0.55) * yaw_rate

    if not pivot and abs(desired_linear) > 1e-6:
        desired_linear *= _yaw_response_speed_scale(desired_yaw, yaw_rate, profile)

    if edge_state is not None and edge_safety is not None and edge_safety["caution"]:
        desired_linear = _limit_outward_linear(
            desired_linear,
            edge_state["forward_outward_cos"],
            edge_safety["safe_outward_speed"],
        )
        edge_yaw_scale = 1.0 - edge_attack_commit * profile.get("edge_attack_yaw_relief", 0.85)
        desired_yaw += (
            profile.get("edge_yaw_k", 3.2)
            * edge_safety["risk"]
            * edge_yaw_scale
            * math.tanh(edge_state["escape_bearing"])
        )
        if edge_safety["rescue"]:
            rescue_bearing = edge_state["escape_bearing"]
            desired_yaw = (
                profile.get("edge_rescue_bearing_k", profile.get("bearing_k", 4.2) + 0.8) * rescue_bearing
                - profile.get("yaw_damp", 0.55) * yaw_rate
            )
            if abs(rescue_bearing) > profile.get("edge_rescue_pivot_angle", 1.10):
                desired_linear = 0.0
            else:
                rescue_speed = min(profile.get("edge_rescue_speed", 1.4), profile.get("max_speed", 1.4))
                desired_linear = edge_state["escape_sign"] * rescue_speed

    desired_yaw = _clip(desired_yaw, -profile.get("max_yaw_rate", 5.0), profile.get("max_yaw_rate", 5.0))

    linear_cmd = _slew_limit(desired_linear, _LAST_LINEAR_CMD, profile.get("linear_slew_up", 0.10))
    if desired_linear < _LAST_LINEAR_CMD:
        linear_cmd = _slew_limit(desired_linear, _LAST_LINEAR_CMD, profile.get("linear_slew_down", 0.16))
    yaw_cmd = _slew_limit(desired_yaw, _LAST_YAW_CMD, profile.get("yaw_slew", 0.24))

    traction = _traction_scale(linear_cmd, metrics["forward_speed"], wheel_omegas)
    if linear_cmd > 0.0:
        linear_cmd *= traction

    if (not ground_contact) or is_tipping > profile.get("tip_cutoff", 0.35):
        linear_cmd = 0.0
        yaw_cmd *= 0.5

    _LAST_T = t
    _LAST_LINEAR_CMD = linear_cmd
    _LAST_YAW_CMD = yaw_cmd

    return linear_cmd, yaw_cmd, metrics, act_vel


def _profile_to_wheels(obs, target_xy, profile, boost_active=False):
    linear_cmd, yaw_cmd, _, act_vel = _compute_body_command(obs, target_xy, profile, boost_active=boost_active)
    wheel_targets, turn_ratio = _wheel_targets_from_body_command(linear_cmd, yaw_cmd, profile)
    max_cmd = profile.get("max_cmd", 0.8)
    max_step = profile.get("max_step", 0.12)
    if boost_active:
        max_cmd = profile.get("boost_max_cmd", max_cmd)
        max_step = profile.get("boost_max_step", max_step)
    return _wheel_speed_control(
        wheel_targets,
        act_vel,
        int(obs.get("t", 0)),
        max_cmd=max_cmd,
        max_step=max_step,
        turn_ratio=turn_ratio,
    )


def _sumo_profile(mode, opp_edge):
    if mode == "recover":
        return {
            "max_speed": 1.2,
            "min_speed": 0.0,
            "max_brake_acc": 8.5,
            "max_lateral_acc": 4.0,
            "stop_radius": 0.20,
            "lookahead_min": 0.55,
            "lookahead_max": 1.3,
            "pivot_angle": 1.0,
            "pivot_distance": 0.25,
            "bearing_k": 4.4,
            "lateral_k": 1.5,
            "yaw_damp": 0.6,
            "pivot_k": 5.8,
            "max_yaw_rate": 5.8,
            "min_align_scale": 0.10,
            "align_heading_gain": 0.9,
            "max_cmd": 0.75,
            "max_step": 0.13,
        }
    if mode == "circle":
        return {
            "max_speed": 2.0,
            "min_speed": 0.55,
            "max_brake_acc": 7.5,
            "max_lateral_acc": 4.8,
            "stop_radius": 0.28,
            "lookahead_min": 0.70,
            "lookahead_max": 1.8,
            "pivot_angle": 1.25,
            "pivot_distance": 0.35,
            "bearing_k": 4.0,
            "lateral_k": 1.2,
            "yaw_damp": 0.50,
            "max_yaw_rate": 5.0,
            "min_align_scale": 0.32,
            "align_heading_gain": 0.65,
            "max_cmd": 0.88,
            "max_step": 0.15,
        }
    if mode == "reposition":
        return {
            "max_speed": 1.9,
            "min_speed": 0.3,
            "max_brake_acc": 7.5,
            "max_lateral_acc": 4.6,
            "stop_radius": 0.22,
            "lookahead_min": 0.65,
            "lookahead_max": 1.7,
            "pivot_angle": 1.15,
            "pivot_distance": 0.35,
            "bearing_k": 4.1,
            "lateral_k": 1.3,
            "yaw_damp": 0.50,
            "max_yaw_rate": 5.2,
            "min_align_scale": 0.20,
            "align_heading_gain": 0.72,
            "max_cmd": 0.86,
            "max_step": 0.15,
        }

    edge_close = opp_edge is not None and float(opp_edge) < 0.9
    return {
        "max_speed": 2.3 if edge_close else 2.1,
        "min_speed": 0.70 if edge_close else 0.55,
        "max_brake_acc": 9.0,
        "max_lateral_acc": 5.4,
        "stop_radius": 0.18,
        "lookahead_min": 0.72,
        "lookahead_max": 1.9,
        "pivot_angle": 1.28,
        "pivot_distance": 0.40,
        "bearing_k": 4.0,
        "lateral_k": 1.0,
        "yaw_damp": 0.48,
        "max_yaw_rate": 5.0,
        "min_align_scale": 0.34,
        "align_heading_gain": 0.60,
        "max_cmd": 0.92,
        "max_step": 0.16,
        "boost_speed_scale": 1.12,
        "boost_max_cmd": 1.0,
        "boost_max_step": 0.20,
    }


def _target_chase_profile(capture_radius, cautious=False, precise=False, rescue=False):
    if rescue:
        return {
            "max_speed": 1.7,
            "min_speed": 0.0,
            "max_brake_acc": 12.0,
            "max_lateral_acc": 4.4,
            "stop_radius": 0.12,
            "lookahead_min": 0.42,
            "lookahead_max": 1.00,
            "pivot_angle": 0.95,
            "pivot_distance": 0.15,
            "bearing_k": 5.0,
            "lateral_k": 1.7,
            "yaw_damp": 0.70,
            "pivot_k": 9.6,
            "max_yaw_rate": 11.0,
            "min_align_scale": 0.04,
            "align_heading_gain": 1.08,
            "max_cmd": 1.0,
            "max_step": 0.22,
            "linear_slew_up": 0.32,
            "linear_slew_down": 0.60,
            "yaw_slew": 0.58,
            "allow_reverse": True,
            "reverse_angle": 1.55,
            "reverse_distance_max": 4.5,
            "edge_brake_acc": 12.0,
            "edge_base_buffer": 0.26,
            "edge_hard_buffer": 0.36,
            "edge_caution_buffer": 1.10,
            "edge_reaction_time": 0.10,
            "edge_turn_time_gain": 0.70,
            "edge_free_bearing": 0.10,
            "edge_yaw_k": 4.0,
            "edge_rescue_speed": 1.6,
            "edge_rescue_pivot_angle": 1.20,
            "edge_rescue_bearing_k": 5.6,
            "turn_gain": 1.22,
            "turn_linear_drop": 0.34,
            "turn_linear_floor": 0.04,
            "turn_diff_boost": 0.26,
            "inside_feather_gain": 0.30,
            "turn_slow_start": 0.22,
            "turn_slow_end": 0.92,
            "turn_slow_floor": 0.02,
            "turn_slow_power": 1.0,
            "yaw_error_slow_start": 0.18,
            "yaw_error_floor": 0.12,
            "yaw_error_gain": 0.85,
        }
    if cautious:
        return {
            "max_speed": 1.3,
            "min_speed": 0.0,
            "max_brake_acc": 9.0,
            "max_lateral_acc": 4.2,
            "stop_radius": 0.16,
            "lookahead_min": 0.55,
            "lookahead_max": 1.2,
            "pivot_angle": 1.0,
            "pivot_distance": 0.25,
            "bearing_k": 4.6,
            "lateral_k": 1.6,
            "yaw_damp": 0.62,
            "pivot_k": 8.0,
            "max_yaw_rate": 9.0,
            "min_align_scale": 0.08,
            "align_heading_gain": 1.02,
            "max_cmd": 0.82,
            "max_step": 0.14,
            "linear_slew_up": 0.18,
            "linear_slew_down": 0.42,
            "yaw_slew": 0.42,
            "allow_reverse": True,
            "reverse_angle": 1.75,
            "reverse_distance_max": 2.0,
            "edge_brake_acc": 10.0,
            "edge_base_buffer": 0.24,
            "edge_hard_buffer": 0.34,
            "edge_caution_buffer": 1.05,
            "edge_reaction_time": 0.10,
            "edge_turn_time_gain": 0.68,
            "edge_yaw_k": 3.8,
            "edge_rescue_speed": 1.1,
            "turn_gain": 1.16,
            "turn_linear_drop": 0.22,
            "turn_linear_floor": 0.14,
            "turn_diff_boost": 0.18,
            "inside_feather_gain": 0.20,
            "turn_slow_start": 0.26,
            "turn_slow_end": 0.98,
            "turn_slow_floor": 0.06,
            "turn_slow_power": 1.1,
            "yaw_error_slow_start": 0.22,
            "yaw_error_floor": 0.14,
            "yaw_error_gain": 0.80,
        }
    if precise:
        return {
            "max_speed": 2.0,
            "min_speed": 0.22,
            "max_brake_acc": 10.0,
            "max_lateral_acc": 5.0,
            "stop_radius": max(0.05, capture_radius * 0.18),
            "lookahead_min": 0.50,
            "lookahead_max": 1.15,
            "pivot_angle": 1.02,
            "pivot_distance": 0.22,
            "bearing_k": 4.8,
            "lateral_k": 1.4,
            "yaw_damp": 0.58,
            "pivot_k": 8.4,
            "max_yaw_rate": 9.5,
            "min_align_scale": 0.12,
            "align_heading_gain": 0.92,
            "max_cmd": 0.96,
            "max_step": 0.18,
            "linear_slew_up": 0.22,
            "linear_slew_down": 0.40,
            "yaw_slew": 0.40,
            "allow_reverse": True,
            "reverse_angle": 1.80,
            "reverse_distance_max": 1.8,
            "edge_brake_acc": 10.5,
            "edge_base_buffer": 0.24,
            "edge_hard_buffer": 0.34,
            "edge_caution_buffer": 1.05,
            "edge_reaction_time": 0.09,
            "edge_turn_time_gain": 0.62,
            "edge_yaw_k": 3.6,
            "edge_rescue_speed": 1.2,
            "turn_gain": 1.12,
            "turn_linear_drop": 0.16,
            "turn_linear_floor": 0.20,
            "turn_diff_boost": 0.14,
            "inside_feather_gain": 0.16,
            "turn_slow_start": 0.34,
            "turn_slow_end": 1.00,
            "turn_slow_floor": 0.10,
            "turn_slow_power": 1.2,
            "yaw_error_slow_start": 0.24,
            "yaw_error_floor": 0.18,
            "yaw_error_gain": 0.76,
        }
    return {
        "max_speed": 3.1,
        "min_speed": 0.85,
        "max_brake_acc": 10.0,
        "max_lateral_acc": 6.6,
        "stop_radius": max(0.06, capture_radius * 0.22),
        "lookahead_min": 0.68,
        "lookahead_max": 1.9,
        "lookahead_speed_gain": 0.22,
        "max_curvature": 3.8,
        "pivot_angle": 1.22,
        "pivot_distance": 0.35,
        "bearing_k": 4.2,
        "lateral_k": 1.2,
        "yaw_damp": 0.46,
        "pivot_k": 8.2,
        "max_yaw_rate": 10.5,
        "min_align_scale": 0.48,
        "align_heading_gain": 0.78,
        "keep_moving_angle": 1.15,
        "max_cmd": 1.0,
        "max_step": 0.20,
        "linear_slew_up": 0.24,
        "linear_slew_down": 0.42,
        "yaw_slew": 0.36,
        "allow_reverse": True,
        "reverse_angle": 1.72,
        "reverse_distance_max": 2.5,
        "edge_brake_acc": 11.0,
        "edge_base_buffer": 0.24,
        "edge_hard_buffer": 0.34,
        "edge_caution_buffer": 1.12,
        "edge_reaction_time": 0.09,
        "edge_turn_time_gain": 0.60,
        "edge_yaw_k": 3.4,
        "edge_rescue_speed": 1.4,
        "turn_gain": 1.08,
        "turn_linear_drop": 0.10,
        "turn_linear_floor": 0.26,
        "turn_diff_boost": 0.10,
        "inside_feather_gain": 0.12,
        "turn_slow_start": 0.42,
        "turn_slow_end": 1.02,
        "turn_slow_floor": 0.18,
        "turn_slow_power": 1.25,
        "yaw_error_slow_start": 0.26,
        "yaw_error_floor": 0.22,
        "yaw_error_gain": 0.70,
    }


def _profile_with_updates(profile, **updates):
    out = dict(profile)
    out.update(updates)
    return out


def _sequence_profile(capture_radius, cautious=False, precise=False, rescue=False):
    if rescue:
        return _profile_with_updates(
            _target_chase_profile(capture_radius, rescue=True),
            stop_radius=0.10,
            edge_rescue_speed=1.7,
            turn_gain=1.26,
            turn_diff_boost=0.30,
        )
    if cautious:
        return _profile_with_updates(
            _target_chase_profile(capture_radius, cautious=True),
            min_speed=0.35,
            stop_radius=max(0.04, capture_radius * 0.10),
            keep_moving_angle=1.24,
            turn_gain=1.20,
            turn_diff_boost=0.24,
            lookahead_speed_gain=0.20,
        )

    base = _target_chase_profile(capture_radius, precise=precise)
    return _profile_with_updates(
        base,
        min_speed=0.70 if precise else 1.15,
        stop_radius=max(0.03, capture_radius * 0.08),
        keep_moving_angle=1.28,
        lookahead_max=max(1.9, base.get("lookahead_max", 1.9)),
        lookahead_speed_gain=max(0.24, base.get("lookahead_speed_gain", 0.22)),
        turn_gain=base.get("turn_gain", 1.0) + 0.05,
        turn_diff_boost=base.get("turn_diff_boost", 0.0) + 0.06,
        turn_slow_start=max(0.18, base.get("turn_slow_start", 0.42) - 0.06),
        yaw_error_slow_start=max(0.16, base.get("yaw_error_slow_start", 0.26) - 0.04),
    )


def _single_player_target_wheels(
    obs,
    *,
    aim_target_xy,
    capture_radius,
    edge_margin,
    target_distance,
    profile_builder,
):
    global _EDGE_RECOVERY_LATCH

    edge_state = _edge_state(obs, allow_reverse=True)
    precise = target_distance < capture_radius + 0.45
    normal_profile = profile_builder(capture_radius, precise=precise)
    edge_safety = _edge_safety(edge_state, normal_profile)

    release_margin = max(1.35, edge_margin * 1.45)
    if _EDGE_RECOVERY_LATCH and (
        edge_state is None
        or (
            edge_state["edge_distance"] >= release_margin
            and edge_state["radial_speed_out"] <= 0.05
        )
    ):
        _EDGE_RECOVERY_LATCH = False

    if edge_safety is not None and (
        edge_safety["rescue"]
        or (
            edge_safety["caution"]
            and (
                edge_safety["risk"] > 0.42
                or (
                    edge_state is not None
                    and edge_state["radial_speed_out"] > edge_safety["safe_outward_speed"] + 0.05
                )
            )
        )
    ):
        _EDGE_RECOVERY_LATCH = True

    if _EDGE_RECOVERY_LATCH:
        target_xy = (0.0, 0.0)
        profile = profile_builder(capture_radius, rescue=True)
    else:
        target_xy = (float(aim_target_xy[0]), float(aim_target_xy[1]))
        if edge_safety is not None and edge_safety["caution"]:
            center_blend = 0.25 + 0.45 * edge_safety["risk"]
            target_xy = (
                float(target_xy[0]) * (1.0 - center_blend),
                float(target_xy[1]) * (1.0 - center_blend),
            )
            profile = profile_builder(capture_radius, cautious=edge_safety["risk"] > 0.45)
        else:
            profile = normal_profile

    return _profile_to_wheels(obs, target_xy, profile, boost_active=False)


def _sequence_target_xy(game, capture_radius):
    target_pos = game.get("target_pos", [0.0, 0.0, 0.0])
    if len(target_pos) < 2:
        return (0.0, 0.0)

    current_xy = (float(target_pos[0]), float(target_pos[1]))
    next_target = game.get("next_target_pos", [])
    if len(next_target) < 2:
        return current_xy

    target_distance = float(game.get("target_distance", 0.0))
    blend_start = max(capture_radius + 0.70, 1.45)
    if target_distance >= blend_start:
        return current_xy

    alpha = _clip(
        (blend_start - target_distance) / max(1e-6, blend_start - capture_radius),
        0.0,
        0.62,
    )
    next_xy = (float(next_target[0]), float(next_target[1]))
    return (
        current_xy[0] * (1.0 - alpha) + next_xy[0] * alpha,
        current_xy[1] * (1.0 - alpha) + next_xy[1] * alpha,
    )


def _update_push_boost(mode, opp_contact, dist_opp, metrics, t):
    global _DEADLOCK_COUNT, _BOOST_UNTIL

    deadlock = (
        mode == "push"
        and (opp_contact or dist_opp < 0.7)
        and abs(metrics["forward_speed"]) < 0.05
        and metrics["speed_xy"] < 0.08
    )
    if deadlock:
        _DEADLOCK_COUNT += 1
        if _DEADLOCK_COUNT >= 20:
            _BOOST_UNTIL = max(_BOOST_UNTIL, t + 40)
    else:
        _DEADLOCK_COUNT = max(0, _DEADLOCK_COUNT - 1)

    return mode == "push" and t < _BOOST_UNTIL


def _play_target_chase(obs, game):
    target_pos = game.get("target_pos", [0.0, 0.0, 0.0])
    if len(target_pos) < 2:
        return dict(_ZERO_WHEEL_MAP)

    capture_radius = float(game.get("capture_radius", 0.5))
    edge_margin = float(game.get("edge_margin", 0.9))
    target_distance = float(game.get("target_distance", 0.0))
    return _single_player_target_wheels(
        obs,
        aim_target_xy=(float(target_pos[0]), float(target_pos[1])),
        capture_radius=capture_radius,
        edge_margin=edge_margin,
        target_distance=target_distance,
        profile_builder=_target_chase_profile,
    )


def _play_waypoint_sequence(obs, game):
    target_pos = game.get("target_pos", [0.0, 0.0, 0.0])
    if len(target_pos) < 2:
        return dict(_ZERO_WHEEL_MAP)

    capture_radius = float(game.get("capture_radius", 0.58))
    edge_margin = float(game.get("edge_margin", 0.9))
    target_distance = float(game.get("target_distance", 0.0))
    aim_target_xy = _sequence_target_xy(game, capture_radius)
    return _single_player_target_wheels(
        obs,
        aim_target_xy=aim_target_xy,
        capture_radius=capture_radius,
        edge_margin=edge_margin,
        target_distance=target_distance,
        profile_builder=_sequence_profile,
    )


def _play_sumo(obs):
    plan = _sumo_waypoint_plan(obs)
    first_waypoint = plan["waypoints"][0] if plan["waypoints"] else (0.0, 0.0)
    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    first_distance = math.hypot(
        float(first_waypoint[0]) - float(my_pos[0]),
        float(first_waypoint[1]) - float(my_pos[1]),
    )
    profile = _sumo_waypoint_profile(plan["mode"], first_distance)
    _set_policy_debug(
        {
            "mode": plan["mode"],
            "orbit_direction": plan["orbit_direction"],
            "orbit_radius": float(plan["orbit_radius"]),
            "ring_radius": float(plan["ring_radius"]),
            "opponent_distance": float(plan["opponent_distance"]),
            "offense_attempt": int(plan.get("offense_attempt", 0)),
            "stall_path_length": float(plan.get("stall_path_length", 0.0)),
            "offense_reason": str(plan.get("offense_reason", "")),
            "late_match_force": bool(plan.get("late_match_force", False)),
            "waypoints": [
                [float(point[0]), float(point[1]), 0.06]
                for point in plan["waypoints"][:4]
            ],
        }
    )
    return _profile_to_wheels(
        obs,
        plan["aim_target_xy"],
        profile,
        boost_active=bool(plan.get("boost_active", False)),
    )


def policy_step(obs) -> dict[str, float]:
    global _LAST_T

    t = int(obs.get("t", 0))
    if _LAST_T is not None and (t < _LAST_T or t == 0):
        _reset_controller_state()

    _set_policy_debug({})

    game = obs.get("game", {}) or {}
    if game.get("mode") == "target-chase" and game.get("active", False):
        return _play_target_chase(obs, game)
    if game.get("mode") == "waypoint-sequence" and game.get("active", False):
        return _play_waypoint_sequence(obs, game)
    return _play_sumo(obs)
