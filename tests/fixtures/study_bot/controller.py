def policy_step(obs) -> dict[str, float]:
    # LeashWedge Center Patrol controller.
    # Main fix from prior failures: never chase the opponent all the way to the lip.
    # The bot patrols the center, attacks only from safe interior positions, and
    # uses velocity feedback to prevent runaway speed.
    import math

    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    def wrap(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    my_pos = obs.get('my_pos', [0.0, 0.0, 0.0])
    opp_pos = obs.get('opponent_pos', [0.0, 0.0, 0.0])
    my_vel = obs.get('my_velocity', [0.0, 0.0, 0.0])
    opp_vel = obs.get('opponent_velocity', [0.0, 0.0, 0.0])

    x = float(my_pos[0])
    y = float(my_pos[1])
    ox = float(opp_pos[0])
    oy = float(opp_pos[1])
    vx = float(my_vel[0])
    vy = float(my_vel[1])

    yaw = float(obs.get('my_yaw', 0.0))
    t = int(obs.get('t', 0))
    my_edge = float(obs.get('my_edge_distance', 99.0))
    opp_edge = float(obs.get('opponent_edge_distance', 99.0))
    dist = float(obs.get('distance_to_opponent', 99.0))
    contact = bool(obs.get('opponent_contact', False))
    grounded = bool(obs.get('ground_contact', True))
    tipping = float(obs.get('is_tipping', 0.0))
    inactivity = float(obs.get('my_inactivity_timer', 0.0))

    # Stop wheel spin while airborne; this avoids adding pitch or roll energy.
    if not grounded:
        return {
            'motor_fl': 0.0,
            'motor_ml': 0.0,
            'motor_rl': 0.0,
            'motor_fr': 0.0,
            'motor_mr': 0.0,
            'motor_rr': 0.0,
        }

    r = math.sqrt(x * x + y * y)
    speed = math.sqrt(vx * vx + vy * vy)
    forward_v = vx * math.cos(yaw) + vy * math.sin(yaw)

    # Radial outward velocity is used for edge risk detection.
    radial_v = 0.0
    if r > 0.05:
        radial_v = (x * vx + y * vy) / r

    # Default behavior is a slow orbit near the center, not direct full-arena pursuit.
    mode = 'orbit'

    # Build a moving patrol target on a safe central circle.
    # If outside the patrol radius, aim inward; otherwise aim tangentially to keep moving.
    if r > 1.8:
        target_x = 0.0
        target_y = 0.0
        base_power = 0.24
        mode = 'center'
    else:
        angle = math.atan2(y, x) if r > 0.20 else (0.012 * float(t))
        patrol_radius = 1.35
        target_angle = angle + 0.90
        target_x = patrol_radius * math.cos(target_angle)
        target_y = patrol_radius * math.sin(target_angle)
        base_power = 0.17

    # Very conservative attack: only approach if both robots are safely interior.
    # This avoids the previous replay behavior of sprinting past the opponent into the edge.
    if my_edge > 4.0 and opp_edge > 1.5 and dist < 2.7 and tipping < 0.35:
        target_x = ox + 0.12 * float(opp_vel[0])
        target_y = oy + 0.12 * float(opp_vel[1])
        base_power = 0.26
        mode = 'attack'

    # If already in contact and I still have plenty of edge margin, continue pushing through.
    if contact and my_edge > 3.2 and tipping < 0.45:
        target_x = ox
        target_y = oy
        base_power = 0.30
        mode = 'shove'

    # Edge leash overrides everything. Trigger much earlier than previous attempts.
    if my_edge < 3.8 or r > 3.7:
        target_x = 0.0
        target_y = 0.0
        base_power = 0.26
        mode = 'retreat'

    # Emergency lip behavior: focus entirely on center recovery with low turn authority.
    if my_edge < 2.2 or (my_edge < 3.2 and radial_v > 0.25):
        target_x = 0.0
        target_y = 0.0
        base_power = 0.22
        mode = 'emergency_retreat'

    target_angle = math.atan2(target_y - y, target_x - x)
    err = wrap(target_angle - yaw)

    # Convert heading error into a smooth drive command.
    # cos(err) means if the target is behind, reverse gently instead of spinning at the edge.
    drive = base_power * math.cos(err)
    if abs(err) > 1.35:
        drive *= 0.55

    turn_gain = 0.24 if mode in ('retreat', 'emergency_retreat') else 0.30
    turn = clamp(turn_gain * err, -0.34, 0.34)

    # Strong velocity governor. This is the key anti-ring-out change.
    # Brake local forward speed if the chassis starts coasting too quickly.
    if speed > 1.05:
        drive -= 0.16 * forward_v
        turn *= 0.80
    if speed > 1.55:
        drive -= 0.28 * forward_v
        turn *= 0.55
    if speed > 2.10:
        drive -= 0.40 * forward_v
        turn *= 0.35

    # If moving outward near the edge, further suppress power.
    if my_edge < 3.5 and radial_v > 0.10:
        drive *= 0.65
        turn *= 0.75

    # Tipping recovery: stop attacking and gently head to the center.
    if tipping > 0.55:
        center_angle = math.atan2(-y, -x)
        center_err = wrap(center_angle - yaw)
        drive = 0.12 * math.cos(center_err)
        turn = clamp(0.20 * center_err, -0.25, 0.25)

    # Startup ramp to prevent initial launch or yaw overshoot.
    if t < 140:
        ramp = 0.25 + 0.75 * (float(t) / 140.0)
        drive *= ramp
        turn *= ramp

    # Inactivity prevention: if somehow nearly stalled in the safe interior,
    # add a small alternating turn and forward crawl.
    if inactivity > 3.0 and my_edge > 3.0 and speed < 0.20:
        wiggle = 0.10 if ((t // 60) % 2 == 0) else -0.10
        turn = clamp(turn + wiggle, -0.25, 0.25)
        if abs(drive) < 0.12:
            drive = 0.14

    # Final caps are tighter near the edge.
    if my_edge < 2.5:
        drive = clamp(drive, -0.24, 0.24)
        turn = clamp(turn, -0.22, 0.22)
    else:
        drive = clamp(drive, -0.36, 0.36)
        turn = clamp(turn, -0.34, 0.34)

    # Differential drive mixing. Positive both sides drives local +X.
    left = clamp(drive - turn, -1.0, 1.0)
    right = clamp(drive + turn, -1.0, 1.0)

    return {
        'motor_fl': left,
        'motor_ml': left,
        'motor_rl': left,
        'motor_fr': right,
        'motor_mr': right,
        'motor_rr': right,
    }