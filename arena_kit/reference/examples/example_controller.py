def policy_step(obs):
    import math
    import numpy as np

    def clamp(x, lo=-1.0, hi=1.0):
        return lo if x < lo else (hi if x > hi else x)

    def wrap_pi(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def slew(desired, prev, rate=0.07):
        d = desired - prev
        if d > rate:
            return prev + rate
        if d < -rate:
            return prev - rate
        return desired

    # Positions / state
    my_pos = obs["my_pos"]
    opp_pos = obs["opponent_pos"]
    x, y = float(my_pos[0]), float(my_pos[1])
    ox, oy = float(opp_pos[0]), float(opp_pos[1])

    my_yaw = float(obs["my_yaw"])
    wz = float(obs["my_angular_velocity"][2])
    vx, vy = float(obs["my_velocity"][0]), float(obs["my_velocity"][1])
    speed = math.sqrt(vx * vx + vy * vy)

    my_edge = float(obs["my_edge_distance"])
    opp_edge = float(obs["opponent_edge_distance"])
    dist = float(obs["distance_to_opponent"])
    ground = bool(obs["ground_contact"])
    contact = bool(obs.get("opponent_contact", False))

    my_r = float(obs.get("my_bounding_radius", 0.6))
    opp_r = float(obs.get("opponent_bounding_radius", 0.6))

    # Previous actions for slew limiting
    prev = {"drive_FL": 0.0, "drive_FR": 0.0, "drive_RL": 0.0, "drive_RR": 0.0}
    ah = obs.get("action_history", [])
    if ah:
        last = ah[-1]
        for k in prev:
            prev[k] = float(last.get(k, 0.0))

    # If airborne, don't spin up
    if not ground:
        return {k: 0.0 for k in prev}

    # Radial velocity (outward positive)
    r = math.sqrt(x * x + y * y)
    if r > 1e-6:
        v_rad = (x * vx + y * vy) / r
    else:
        v_rad = 0.0

    # Predict edge margin (simple lookahead)
    lookahead = 0.7
    predicted_edge = my_edge - max(0.0, v_rad) * lookahead

    # Edge triggers (slightly less conservative than commit 3)
    edge_danger = (my_edge < 0.9) or (predicted_edge < 0.6)

    t = int(obs["t"])
    early = t < 40  # 0.4s

    # Decide mode
    close_enough = dist < (my_r + opp_r + 0.25)
    in_contact_mode = contact or close_enough

    to_ctr = np.array([-x, -y], dtype=float)
    to_opp = np.array([ox - x, oy - y], dtype=float)

    base = 0.0
    turn = 0.0
    vmax = 2.5

    if edge_danger or early:
        # ESCAPE: go to center
        v = to_ctr
        target_yaw = math.atan2(float(v[1]), float(v[0]))
        ang_err = wrap_pi(target_yaw - my_yaw)

        # Reverse-pull if facing away from center
        if abs(ang_err) > 1.3:
            base = -0.60
        else:
            base = 0.75

        turn = clamp(1.9 * ang_err - 0.25 * wz, -1.0, 1.0)
        vmax = 1.8 if my_edge < 1.2 else 2.2

    else:
        # ATTACK / CONTACT LOGIC
        # Center-side approach: aim for a point slightly inward from the opponent.
        u = np.array([ox, oy], dtype=float)
        un = float(np.linalg.norm(u))
        if un > 1e-6:
            u /= un
        else:
            u = np.array([1.0, 0.0], dtype=float)

        # k shrinks if opponent is already near edge (avoid over-committing outward)
        if opp_edge < 1.5:
            k = 0.6
        else:
            k = 1.0

        aim = np.array([ox, oy], dtype=float) - k * u
        v = aim - np.array([x, y], dtype=float)

        # small center bias to prevent outward spirals
        v = v + 0.20 * to_ctr

        if float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) < 1e-9:
            v = to_ctr

        target_yaw = math.atan2(float(v[1]), float(v[0]))
        ang_err = wrap_pi(target_yaw - my_yaw)

        # Base throttle by distance
        if dist > 6.0:
            base = 0.85
        elif dist > 2.5:
            base = 0.65
        else:
            base = 0.50

        turn = clamp(1.35 * ang_err - 0.22 * wz, -1.0, 1.0)

        # Contact commit: if we are safer (more inside) than opponent, push harder
        if in_contact_mode:
            safer = (my_edge > opp_edge + 0.2) and (my_edge > 1.1)
            if safer:
                base = 0.95
                turn = clamp(0.9 * ang_err - 0.15 * wz, -0.7, 0.7)
                vmax = 2.8
            else:
                # Reposition: don't full-send if it risks self ring-out
                base = 0.55
                vmax = 2.2

        # If huge turn needed, partial reverse to avoid wide outward arcs
        if abs(ang_err) > 2.4 and dist > 1.0:
            base = -0.40

        # Speed cap tighter as we approach edge
        if my_edge > 2.8:
            vmax = min(vmax, 3.2)
        elif my_edge > 1.8:
            vmax = min(vmax, 2.6)
        else:
            vmax = min(vmax, 2.1)

    # Speed limiting
    if vmax > 0.1:
        factor = clamp((vmax - speed) / vmax + 0.25, 0.25, 1.0)
        base *= factor

    # Anti-inactivity (only when reasonably safe)
    inact = float(obs["my_inactivity_timer"])
    if inact > 7.0 and my_edge > 1.5 and (not edge_danger):
        phase = (t // 35) % 2
        if phase == 0:
            base = 0.40
            turn = 0.75
        else:
            base = -0.30
            turn = -0.75

    # Convert to left/right commands then map to 4 wheels
    left = clamp(base - turn, -1.0, 1.0)
    right = clamp(base + turn, -1.0, 1.0)

    des = {
        "drive_FL": left,
        "drive_RL": left,
        "drive_FR": right,
        "drive_RR": right,
    }

    out = {}
    for k in des:
        out[k] = float(clamp(slew(des[k], prev[k], rate=0.07), -1.0, 1.0))
    return out