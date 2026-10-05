"""Proportional-navigation policy with attack mode for the pusher dummy.

Drives toward the opponent using differential steering.  On physical
contact, switches to full-power attack.  If stalled, briefly retreats
then re-approaches.

State machine: approach → attack → retreat → approach → ...

Actuator names match pusher_dummy_3d.xml: m_fl, m_fr, m_rl, m_rr.
"""

import math

_state = {"mode": "approach", "timer": 0}


def policy_step(obs) -> dict[str, float]:
    t = int(obs.get("t", 0))
    if t == 0:
        _state["mode"] = "approach"
        _state["timer"] = 0

    my_pos = obs.get("my_pos", [0.0, 0.0, 0.0])
    opp_pos = obs.get("opponent_pos", [0.0, 0.0, 0.0])
    my_yaw = float(obs.get("my_yaw", 0.0))
    my_vel = obs.get("my_velocity", [0.0, 0.0, 0.0])
    opponent_contact = bool(obs.get("opponent_contact", False))
    edge_dist = float(obs.get("my_edge_distance", 10.0))
    ring_radius = float(obs.get("ring_radius", 5.0))

    px, py = float(my_pos[0]), float(my_pos[1])
    ox, oy = float(opp_pos[0]), float(opp_pos[1])
    dx = ox - px
    dy = oy - py
    dist = math.sqrt(dx * dx + dy * dy)
    angle_to_opp = math.atan2(dy, dx)

    err = (angle_to_opp - my_yaw + math.pi) % (2 * math.pi) - math.pi
    heading_factor = max(0.0, math.cos(err))

    vx, vy = float(my_vel[0]), float(my_vel[1])
    fwd_speed = vx * math.cos(my_yaw) + vy * math.sin(my_yaw)
    speed = math.sqrt(vx * vx + vy * vy)

    my_r = math.sqrt(px * px + py * py)
    opp_r = math.sqrt(ox * ox + oy * oy)

    # Radial speed: positive = moving away from ring center
    speed_outward = (vx * px + vy * py) / my_r if my_r > 0.1 else 0.0

    # Edge danger zone scales with speed
    edge_danger = 1.5 + speed * 0.5

    # ── State transitions ─────────────────────────────────────
    mode = _state["mode"]
    _state["timer"] += 1

    if mode == "approach":
        if opponent_contact or dist < 1.2:
            _state["mode"] = "attack"
            _state["timer"] = 0
    elif mode == "attack":
        # Abort attack if near edge
        if edge_dist < edge_danger:
            _state["mode"] = "retreat"
            _state["timer"] = 0
        # Stall detection: 25 steps without progress
        elif _state["timer"] > 25 and abs(fwd_speed) < 0.3:
            _state["mode"] = "retreat"
            _state["timer"] = 0
        # Lost contact and drifted away
        elif not opponent_contact and dist > 1.5:
            _state["mode"] = "approach"
            _state["timer"] = 0
    elif mode == "retreat":
        if _state["timer"] > 5:
            _state["mode"] = "approach"
            _state["timer"] = 0

    mode = _state["mode"]

    # ── Throttle & Steering ──────────────────────────────────
    # Default steering: track opponent
    turn = max(-0.8, min(0.8, 1.2 * err))

    # Priority 1: Edge safety — drive toward center when near edge
    if edge_dist < edge_danger and speed_outward > 0.5:
        angle_to_center = math.atan2(-py, -px)
        center_err = (angle_to_center - my_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(center_err) < math.pi / 2:
            throttle = 0.9
        else:
            throttle = -0.9
        turn = max(-1.0, min(1.0, 1.5 * center_err))

    elif mode == "attack":
        throttle = 1.0 * heading_factor

    elif mode == "retreat":
        # Short reverse to break contact, then back to approach
        if my_r > ring_radius * 0.6:
            # Near edge: drive toward center instead of just reversing
            angle_to_center = math.atan2(-py, -px)
            center_err = (angle_to_center - my_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(center_err) < math.pi / 2:
                throttle = 0.6
            else:
                throttle = -0.6
            turn = max(-1.0, min(1.0, 1.5 * center_err))
        else:
            throttle = -0.4

    else:
        # Approach mode — two-phase:
        #   Close range (dist < 2.5): aggressive push to overcome friction
        #   Far range: speed controller with throttle floor
        if dist < 2.5:
            # Close range: strong constant push to reach/maintain contact
            throttle = 0.9 * heading_factor
        else:
            target_speed = max(1.5, 0.6 * dist)
            speed_err = target_speed - fwd_speed
            sc_throttle = max(-0.3, min(0.5, 0.8 * speed_err))
            throttle = max(0.3, sc_throttle) * heading_factor

    left = max(-1.0, min(1.0, throttle - turn))
    right = max(-1.0, min(1.0, throttle + turn))

    return {
        "m_fl": left,
        "m_fr": right,
        "m_rl": left,
        "m_rr": right,
    }
