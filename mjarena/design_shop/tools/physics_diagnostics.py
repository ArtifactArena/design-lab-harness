"""Extract static physics properties from a compiled MuJoCo model."""

from __future__ import annotations

import mujoco
import numpy as np


def get_physics_diagnostics(
    model: mujoco.MjModel,
    prefix: str = "red_",
    motor_mass_per_gear: float = 0.01,
) -> dict:
    """Return a dict of physics diagnostics for the robot identified by *prefix*.

    All data comes from static ``MjModel`` properties — no simulation needed.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _name(obj_type: int, obj_id: int) -> str:
        return mujoco.mj_id2name(model, obj_type, obj_id) or ""

    def _strip(name: str) -> str:
        return name.removeprefix(prefix)

    OBJ_BODY = mujoco.mjtObj.mjOBJ_BODY
    OBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR
    OBJ_JOINT = mujoco.mjtObj.mjOBJ_JOINT

    # ------------------------------------------------------------------
    # Collect prefixed body IDs
    # ------------------------------------------------------------------
    body_ids: list[int] = []
    for bid in range(model.nbody):
        bname = _name(OBJ_BODY, bid)
        if bname.startswith(prefix):
            body_ids.append(bid)

    # ------------------------------------------------------------------
    # Per-body info  (indexed by body id for quick lookup)
    # ------------------------------------------------------------------
    body_actuator_count: dict[int, int] = {}
    body_total_gear: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Per-actuator info
    # ------------------------------------------------------------------
    actuators: list[dict] = []
    total_gear = 0.0

    for i in range(model.nu):
        act_name = _name(OBJ_ACTUATOR, i)
        if not act_name.startswith(prefix):
            continue

        gear_mag = float(np.linalg.norm(model.actuator_gear[i]))
        total_gear += gear_mag

        # Target IDs belong to the transmission's namespace, not always to
        # joints. Treating a site/tendon ID as a joint can select the other
        # robot's body or crash while reporting a QACC warning.
        target = int(model.actuator_trnid[i, 0])
        transmission = int(model.actuator_trntype[i])
        jnt_id = -1
        if transmission in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            jnt_id = target
            body_id = int(model.jnt_bodyid[jnt_id])
        elif transmission == int(mujoco.mjtTrn.mjTRN_SITE):
            body_id = int(model.site_bodyid[target])
        elif transmission == int(mujoco.mjtTrn.mjTRN_SLIDERCRANK):
            body_id = int(model.site_bodyid[int(model.actuator_trnid[i, 1])])
        else:
            # Tendon motors mount on the root assembly. This is a static
            # diagnostic approximation, not effective mechanism inertia.
            body_id = next(b for b in body_ids if int(model.body_parentid[b]) == 0)
        body_name = _name(OBJ_BODY, body_id)
        body_mass = float(model.body_mass[body_id])
        body_inertia = model.body_inertia[body_id].tolist()  # [Ixx, Iyy, Izz]
        body_inertia_min = float(np.min(model.body_inertia[body_id]))

        ratio = gear_mag / body_inertia_min if body_inertia_min > 0 else float("inf")
        max_torque = gear_mag * 1.0  # at ctrl=1
        max_angular_accel = max_torque / body_inertia_min if body_inertia_min > 0 else float("inf")

        actuators.append({
            "name": _strip(act_name),
            "gear": round(gear_mag, 4),
            "joint_id": jnt_id,
            "body_name": _strip(body_name),
            "body_mass_kg": round(body_mass, 4),
            "body_inertia": [round(v, 6) for v in body_inertia],
            "body_inertia_min": round(body_inertia_min, 6),
            "gear_to_inertia_ratio": round(ratio, 1),
            "max_torque_nm": round(max_torque, 4),
            "max_angular_accel": round(max_angular_accel, 1),
        })

        # Track per-body actuator stats
        body_actuator_count[body_id] = body_actuator_count.get(body_id, 0) + 1
        body_total_gear[body_id] = body_total_gear.get(body_id, 0.0) + gear_mag

    # ------------------------------------------------------------------
    # Aggregate mass
    # ------------------------------------------------------------------
    motor_mass = total_gear * motor_mass_per_gear
    geometry_mass = sum(float(model.body_mass[bid]) for bid in body_ids)
    total_mass = geometry_mass  # compiled mass already includes injected motor mass

    # ------------------------------------------------------------------
    # Per-body output
    # ------------------------------------------------------------------
    bodies: list[dict] = []
    for bid in body_ids:
        bname = _name(OBJ_BODY, bid)
        bodies.append({
            "name": _strip(bname),
            "mass_kg": round(float(model.body_mass[bid]), 4),
            "inertia": [round(float(v), 6) for v in model.body_inertia[bid]],
            "pos": [round(float(v), 4) for v in model.body_pos[bid]],
            "num_actuators": body_actuator_count.get(bid, 0),
            "total_gear": round(body_total_gear.get(bid, 0.0), 4),
        })

    # ------------------------------------------------------------------
    # Worst actuator (highest gear-to-inertia ratio)
    # ------------------------------------------------------------------
    if actuators:
        worst = max(actuators, key=lambda a: a["gear_to_inertia_ratio"])
        max_gear_to_inertia_ratio = worst["gear_to_inertia_ratio"]
        worst_actuator = worst["name"]
    else:
        max_gear_to_inertia_ratio = 0.0
        worst_actuator = ""

    # ------------------------------------------------------------------
    # Traction analysis
    # ------------------------------------------------------------------
    total_weight_n = total_mass * 9.81
    max_drive_force_n = total_gear  # worst case: all gear driving linearly

    # Per-actuator friction: find friction of geoms on the driven body
    per_actuator_traction: list[dict] = []
    for act in actuators:
        # Find body id from name (re-derive since we stripped prefix)
        body_full_name = prefix + act["body_name"]
        body_id = mujoco.mj_name2id(model, OBJ_BODY, body_full_name)
        # Get max friction from geoms on this body
        body_friction = 0.0
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) == body_id:
                # friction[0] is sliding friction
                body_friction = max(body_friction, float(model.geom_friction[gid, 0]))
        per_actuator_traction.append({
            "name": act["name"],
            "gear_force_n": round(act["gear"], 4),
            "body_friction": round(body_friction, 4),
        })

    traction = {
        "total_weight_n": round(total_weight_n, 2),
        "max_drive_force_n": round(max_drive_force_n, 2),
        "force_to_weight_ratio": round(max_drive_force_n / total_weight_n, 4) if total_weight_n > 0 else 0.0,
        "per_actuator": per_actuator_traction,
    }

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return {
        # Aggregate
        "total_mass_kg": round(total_mass, 4),
        "geometry_mass_kg": round(geometry_mass, 4),
        "motor_mass_kg": round(motor_mass, 4),
        "total_gear": round(total_gear, 4),
        "num_actuators": len(actuators),
        "num_bodies": len(body_ids),
        "max_gear_to_inertia_ratio": max_gear_to_inertia_ratio,
        "worst_actuator": worst_actuator,
        # Per-actuator
        "actuators": actuators,
        # Per-body
        "bodies": bodies,
        # Traction
        "traction": traction,
    }
