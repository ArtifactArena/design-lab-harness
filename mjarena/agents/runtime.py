"""
Runtime bot execution for DSPy-native arena.
Replaces the legacy contenders module with clean bot abstractions.
"""
from __future__ import annotations

import mujoco
import numpy as np
from collections import deque
from typing import Callable, Dict, List, Mapping
from gymnasium import spaces

from mjarena.agents.types import BotObservation
from mjarena.envs.history import ActionHistory


def _com_for_prefix(
    model: mujoco.MjModel, data: mujoco.MjData, prefix: str, *, is_3d: bool = False
) -> np.ndarray:
    """Mass-weighted COM for all bodies in the contender's physical tree.

    Always returns full 3D position [x, y, z].
    For 2D robots: x = forward/back, y = 0 (locked), z = up.
    For 3D robots: x/y = ground plane, z = up.
    """
    total_m = 0.0
    com = np.zeros(3)
    for i in range(model.nbody):
        if _body_belongs_to_contender(model, i, prefix):
            m = model.body_mass[i]
            total_m += m
            # xipos is the body's world-space COM; xpos is its frame origin.
            com += m * data.xipos[i]
    if total_m <= 0:  # fallback: origin
        return np.zeros(3)
    return com / total_m


def _body_belongs_to_contender(model: mujoco.MjModel, body_id: int, prefix: str) -> bool:
    """Return True for physical robot bodies, excluding team-marker mocaps."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    try:
        if int(model.body_mocapid[body_id]) >= 0:
            return False
    except Exception:
        pass
    if name == f"{prefix}beacon":
        return False
    while body_id > 0 and int(model.body_parentid[body_id]) > 0:
        body_id = int(model.body_parentid[body_id])
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return (root_name.startswith(prefix) and root_name != f"{prefix}beacon"
            and int(model.body_mocapid[body_id]) < 0)


def _find_yaw_joint_qpos_adr(model: mujoco.MjModel, prefix: str) -> int:
    """Find the qpos address for the yaw/hinge_y joint of a robot."""
    for jid in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if jnt_name.startswith(prefix):
            # Look for yaw joint (could be named "yaw", "hinge_y", etc.)
            if "yaw" in jnt_name or "hinge_y" in jnt_name:
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                    return int(model.jnt_qposadr[jid])
    return -1


def _find_freejoint_body_id(model: mujoco.MjModel, prefix: str) -> int:
    """Find the root body with a freejoint for the given prefix. Returns body ID or -1."""
    for bid in range(model.nbody):
        if _body_belongs_to_contender(model, bid, prefix):
            for jid in range(model.njnt):
                if (model.jnt_bodyid[jid] == bid
                        and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
                    return bid
            break
    return -1


def _actuator_ids_for_prefix(model: mujoco.MjModel, prefix: str = None) -> List[int]:
    """Get actuator IDs for a given prefix."""
    ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if prefix is not None:
            if name.startswith(prefix):
                ids.append(i)
        else:
            ids.append(i)
    return ids


def _jsonify_debug_value(value):
    """Convert controller debug values into JSON-safe Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonify_debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_debug_value(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


class BotRuntime:
    """
    DSPy-native bot runtime for match execution.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_callable: Callable[[BotObservation], Mapping[str, float]],
        prefix: str,
        history_len: int = 10,
        obs_lookback: int = 20,
        action_lookback: int = 20,
    ):
        self.model = model
        self.data = data
        self.policy_callable = policy_callable
        self.prefix = prefix
        self.actuator_ids = _actuator_ids_for_prefix(model, prefix)
        self.actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"actuator_{aid}"
            for aid in self.actuator_ids
        ]
        self.action_dim = len(self.actuator_ids)
        self.action_history = ActionHistory(self.action_dim, history_len)
        self._yaw_qpos_adr = _find_yaw_joint_qpos_adr(model, prefix)
        self._freejoint_body_id = _find_freejoint_body_id(model, prefix)
        self._is_3d = self._detect_3d(model, prefix)

        # Rolling history queues exposed to policy via obs_dict
        self._obs_history: deque = deque(maxlen=obs_lookback)
        self._action_history: deque = deque(maxlen=action_lookback)

        # Cache freejoint dof address for velocity readout
        self._freejoint_dof_adr = -1
        if self._freejoint_body_id >= 0:
            for jid in range(model.njnt):
                if (model.jnt_bodyid[jid] == self._freejoint_body_id
                        and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
                    self._freejoint_dof_adr = int(model.jnt_dofadr[jid])
                    break

        # Pre-compute static robot properties (don't change during match)
        self._total_mass = self._compute_total_mass(model, prefix)
        self._bounding_radius = self._compute_bounding_radius(model, data, prefix)

    @staticmethod
    def _detect_3d(model: mujoco.MjModel, prefix: str) -> bool:
        """Return True if the robot with *prefix* uses a freejoint (3D mode)."""
        # Find the root body for this prefix
        for bid in range(model.nbody):
            if _body_belongs_to_contender(model, bid, prefix):
                # Check if any joint on this body is a freejoint
                for jid in range(model.njnt):
                    if (model.jnt_bodyid[jid] == bid
                            and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
                        return True
                break  # only need to check the first (root) body
        return False

    @staticmethod
    def _compute_total_mass(model: mujoco.MjModel, prefix: str) -> float:
        """Sum mass of all bodies belonging to this robot."""
        total = 0.0
        for i in range(model.nbody):
            if _body_belongs_to_contender(model, i, prefix):
                total += float(model.body_mass[i])
        return total

    @staticmethod
    def _compute_bounding_radius(
        model: mujoco.MjModel, data: mujoco.MjData, prefix: str
    ) -> float:
        """Max distance from COM to any geom surface point (XY plane only).

        This gives the 2D "footprint radius" — how far the robot extends
        from its center on the ground plane.  Policies can use it to
        estimate when physical contact will happen:
            contact ≈ dist < my_bounding_radius + opponent_bounding_radius
        """
        # Find body IDs for this robot
        body_ids = set()
        for bid in range(model.nbody):
            if _body_belongs_to_contender(model, bid, prefix):
                body_ids.add(bid)
        if not body_ids:
            return 0.0

        com = _com_for_prefix(model, data, prefix)
        max_r = 0.0
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) not in body_ids:
                continue
            gtype = int(model.geom_type[gid])
            # Skip planes and heightfields
            if gtype in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            # Geom center in world frame
            gpos = data.geom_xpos[gid]
            # Distance from COM to geom center (XY only)
            dx = float(gpos[0] - com[0])
            dy = float(gpos[1] - com[1])
            center_dist = np.sqrt(dx * dx + dy * dy)
            # Add the geom's own bounding sphere radius
            r = center_dist + float(model.geom_rbound[gid])
            if r > max_r:
                max_r = r
        return max_r

    @property
    def total_mass(self) -> float:
        """Total mass of the robot in kg."""
        return self._total_mass

    @property
    def bounding_radius(self) -> float:
        """Max XY extent from COM to any geom surface (meters)."""
        return self._bounding_radius

    @property
    def com_position(self) -> np.ndarray:
        """Get full 3D center of mass position [x, y, z]."""
        return _com_for_prefix(self.model, self.data, self.prefix, is_3d=self._is_3d)

    @property
    def heading(self) -> float:
        """Get robot heading (yaw) in radians. 0 = facing +x direction."""
        return self.yaw

    @property
    def yaw(self) -> float:
        """Get robot yaw in radians. 0 = facing +x direction."""
        if self._freejoint_body_id >= 0:
            xmat = self.data.xmat[self._freejoint_body_id].reshape(3, 3)
            forward_x = xmat[0, 0]
            forward_y = xmat[1, 0]
            return float(np.arctan2(forward_y, forward_x))
        if self._yaw_qpos_adr >= 0:
            return float(self.data.qpos[self._yaw_qpos_adr])
        return 0.0

    @property
    def pitch(self) -> float:
        """Get robot pitch in radians. 0 = level, +/- = tilted forward/back."""
        if self._freejoint_body_id >= 0:
            xmat = self.data.xmat[self._freejoint_body_id].reshape(3, 3)
            return float(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0)))
        return 0.0

    @property
    def roll(self) -> float:
        """Get robot roll in radians. 0 = level, +/- = tilted left/right."""
        if self._freejoint_body_id >= 0:
            xmat = self.data.xmat[self._freejoint_body_id].reshape(3, 3)
            return float(np.arctan2(xmat[2, 1], xmat[2, 2]))
        return 0.0

    @property
    def velocity(self) -> np.ndarray:
        """Get [vx, vy, vz] world-frame linear velocity."""
        if self._freejoint_dof_adr >= 0:
            return np.array(self.data.qvel[self._freejoint_dof_adr:self._freejoint_dof_adr + 3], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Get [wx, wy, wz] world-frame angular velocity."""
        if self._freejoint_dof_adr >= 0:
            return np.array(self.data.qvel[self._freejoint_dof_adr + 3:self._freejoint_dof_adr + 6], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    @property
    def is_tipping(self) -> float:
        """Get tipping metric: 0.0=upright, 1.0=fallen."""
        if self._freejoint_body_id >= 0:
            xmat = self.data.xmat[self._freejoint_body_id].reshape(3, 3)
            tilt_angle = float(np.arccos(np.clip(xmat[2, 2], -1.0, 1.0)))
            return float(np.clip(tilt_angle / (np.pi / 2), 0.0, 1.0))
        return 0.0

    @property
    def action_space(self) -> spaces.Space:
        """Get action space for this bot."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

    @property
    def observation_space(self) -> spaces.Space:
        """Get observation space for this bot (full 3D schema)."""
        return spaces.Dict({
            "my_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "opponent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "my_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "my_pitch": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "my_roll": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "opponent_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "opponent_pitch": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "opponent_roll": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "my_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "my_angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "opponent_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "opponent_angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "distance_to_opponent": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "my_edge_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "opponent_edge_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "opponent_contact": spaces.Discrete(2),
            "opponent_contact_force": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "ground_contact": spaces.Discrete(2),
            "is_tipping": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "max_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
        })

    def act(self, observation: BotObservation) -> np.ndarray:
        """Generate action from observation using the bot's policy."""
        # Convert BotObservation to dict format expected by policies
        obs_dict = {
            # Position
            "my_pos": observation.my_pos,
            "opponent_pos": observation.opponent_pos,
            # Orientation
            "my_yaw": observation.my_yaw,
            "my_pitch": observation.my_pitch,
            "my_roll": observation.my_roll,
            "opponent_yaw": observation.opponent_yaw,
            "opponent_pitch": observation.opponent_pitch,
            "opponent_roll": observation.opponent_roll,
            # Velocity
            "my_velocity": observation.my_velocity,
            "my_angular_velocity": observation.my_angular_velocity,
            "opponent_velocity": observation.opponent_velocity,
            "opponent_angular_velocity": observation.opponent_angular_velocity,
            # Distances
            "distance_to_opponent": observation.distance_to_opponent,
            "my_edge_distance": observation.my_edge_distance,
            "opponent_edge_distance": observation.opponent_edge_distance,
            # Contact
            "opponent_contact": observation.opponent_contact,
            "opponent_contact_force": observation.opponent_contact_force,
            "ground_contact": observation.ground_contact,
            # Stability
            "is_tipping": observation.is_tipping,
            # Time
            "t": observation.t,
            "max_t": observation.max_t,
            # Optional task/game context
            "game": dict(observation.game),
            # Actuator velocities (joint-space, rad/s or m/s)
            "my_actuator_velocity": observation.my_actuator_velocity,
            "opponent_actuator_velocity": observation.opponent_actuator_velocity,
            # Robot properties
            "my_bounding_radius": observation.my_bounding_radius,
            "opponent_bounding_radius": observation.opponent_bounding_radius,
            "ring_radius": observation.ring_radius,
            # Inactivity
            "my_inactivity_timer": observation.my_inactivity_timer,
            "opponent_inactivity_timer": observation.opponent_inactivity_timer,
            # Legacy aliases
            "my_heading": observation.my_yaw,
            "my_closest_distance_to_ring": observation.my_edge_distance,
            # History (rolling queues, most recent last — empty at t=0)
            "obs_history": list(self._obs_history),
            "action_history": list(self._action_history),
        }

        # Spatial grids (not stored in history — too large and rebuilt each step)
        if observation.arena_grid is not None:
            obs_dict["arena_grid"] = observation.arena_grid
        if observation.arena_mass_grid is not None:
            obs_dict["arena_mass_grid"] = observation.arena_mass_grid
        if observation.edge_distance_grid is not None:
            obs_dict["edge_distance_grid"] = observation.edge_distance_grid

        # Detailed observations are detached snapshots, safe for controller state.
        obs_dict.update(observation.details)

        # Get action from policy
        action_dict = self.policy_callable(obs_dict)

        # Convert to numpy array in actuator order
        action_array = np.zeros(self.action_dim, dtype=np.float32)
        for i, aid in enumerate(self.actuator_ids):
            actuator_name = self.actuator_names[i]
            # Remove prefix from actuator name to get base name
            base_name = actuator_name[len(self.prefix):] if self.prefix else actuator_name
            action_array[i] = action_dict.get(base_name, 0.0)

        # Record current obs and action to history for next timestep
        self._obs_history.append({
            "my_pos": observation.my_pos.copy(),
            "opponent_pos": observation.opponent_pos.copy(),
            "my_yaw": float(observation.my_yaw),
            "my_velocity": observation.my_velocity.copy(),
            "opponent_velocity": observation.opponent_velocity.copy(),
            "distance_to_opponent": float(observation.distance_to_opponent),
            "my_edge_distance": float(observation.my_edge_distance),
            "opponent_edge_distance": float(observation.opponent_edge_distance),
            "t": observation.t,
            "my_actuator_velocity": dict(observation.my_actuator_velocity),
            "game": dict(observation.game),
        })
        self._action_history.append(dict(action_dict))

        return action_array

    def actuator_joint_velocity(self, data=None) -> Dict[str, float]:
        """Return per-transmission rates; ball rates follow the motor torque axis.

        Gear magnitude is removed except for relative-site transmissions.
        """
        data = self.data if data is None else data
        velocities: Dict[str, float] = {}
        if self.model is None or data is None:
            return velocities
        for aid, name in zip(self.actuator_ids, self.actuator_names):
            base_name = name[len(self.prefix):] if self.prefix else name
            vel = 0.0
            try:
                jid = int(self.model.actuator_trnid[aid][0])
                transmission = int(self.model.actuator_trntype[aid])
                if transmission == int(mujoco.mjtTrn.mjTRN_TENDON):
                    vel = float(data.ten_velocity[jid])
                elif transmission in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)) and 0 <= jid < self.model.njnt:
                    if int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_BALL):
                        # MuJoCo resolves joint vs jointinparent frames and gear
                        # direction. Normalize only magnitude to obtain rad/s.
                        magnitude = float(np.linalg.norm(self.model.actuator_gear[aid][:3]))
                        vel = float(data.actuator_velocity[aid]) / magnitude if magnitude > 0 else 0.0
                    else:
                        dof_adr = int(self.model.jnt_dofadr[jid])
                        vel = float(data.qvel[dof_adr])
                elif transmission == int(mujoco.mjtTrn.mjTRN_SITE):
                    # Six-component gear has no single scalar scale to divide out.
                    vel = float(data.actuator_velocity[aid])
                else:
                    act_vel = float(getattr(data, "actuator_velocity", [0.0])[aid])
                    gear = float(self.model.actuator_gear[aid][0]) if self.model.actuator_gear.size else 0.0
                    if abs(gear) > 1e-6:
                        vel = act_vel / gear
                    else:
                        vel = act_vel
            except Exception:
                vel = 0.0
            velocities[base_name] = vel
        return velocities

    def controller_debug(self) -> Dict[str, object]:
        """Return optional controller debug payload for telemetry/replay."""
        debug_fn = getattr(self.policy_callable, "debug_payload", None)
        if not callable(debug_fn):
            return {}
        raw = debug_fn()
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(key): _jsonify_debug_value(value)
            for key, value in raw.items()
        }

    def apply_action(self, action: np.ndarray) -> None:
        """Apply action to MuJoCo model."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Action shape {action.shape} != expected {(self.action_dim,)}")
        
        # Clamp actions to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Apply to MuJoCo
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = action[i]
        
        # Update action history
        self.action_history.append(action)
