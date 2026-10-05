# Known-good fixture controller for the six-wheel LeashWedge hardware.
#
# Its only job is to never lose a qualification round under the game's rules: stay on
# the ring, stay stable, keep the centre of mass moving (the inactivity rule loses a bot
# that stays within 0.5 m for 10 s). This chassis barely turns (six wheels, heavy keel),
# so it does not try to: it drives straight strokes of about 2.5 s forward and back,
# each carrying it well over 0.5 m, and reverses early whenever a stroke heads toward
# the lip. It never seeks the opponent.


def policy_step(obs):
    import math

    def clamp(v, lo=-1.0, hi=1.0):
        return lo if v < lo else (hi if v > hi else v)

    x, y = float(obs["my_pos"][0]), float(obs["my_pos"][1])
    vx, vy = float(obs["my_velocity"][0]), float(obs["my_velocity"][1])
    yaw = float(obs["my_yaw"])
    t = int(obs["t"])
    my_edge = float(obs["my_edge_distance"])

    speed = math.hypot(vx, vy)
    forward_v = vx * math.cos(yaw) + vy * math.sin(yaw)
    # Would driving forward take me outward? (+1 if my nose points away from the centre)
    outward = math.cos(yaw) * x + math.sin(yaw) * y
    outward = outward / math.hypot(x, y) if math.hypot(x, y) > 0.05 else 0.0

    stroke = 1.0 if (t // 250) % 2 == 0 else -1.0   # 2.5 s forward, 2.5 s back
    if my_edge < 2.5:                                # edge guard: only move inward
        stroke = -1.0 if outward > 0.0 else 1.0

    drive = 0.34 * stroke
    if speed > 1.05:                                 # governor
        drive -= 0.16 * forward_v
    if t < 100:
        drive *= 0.3 + 0.7 * (float(t) / 100.0)      # startup ramp

    v = clamp(drive, -0.36, 0.36)
    return {"motor_fl": v, "motor_ml": v, "motor_rl": v,
            "motor_fr": v, "motor_mr": v, "motor_rr": v}
