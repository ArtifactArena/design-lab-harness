# Run a single match using your env; adjust imports if env API differs.
import logging
import math
import time
import datetime
import threading
import mujoco
from tqdm import tqdm
import numpy as np
from pathlib import Path

from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, Tuple, Type, Mapping, Union

if TYPE_CHECKING:
    from mjarena.core.build_config import ObservationConfig

from mjarena.envs.sumo import SumoEnv
from mjarena.envs.utils import _mj_load, clamp_actuator_gears
from mjarena.envs.occupancy_grid import GridCache, make_grid_cache, build_arena_grid, build_arena_mass_grid
from mjarena.agents.runtime import BotRuntime
from mjarena.agents.types import BotObservation
from mjarena.design_shop.types import GameRecord
from mjarena.runner.recording import VideoOverlayInfo, capture_frame, format_tqdm_bar, get_offscreen_writer

# Setup logger
logger = logging.getLogger(__name__)


def _trace(message: str, *, enabled: bool) -> None:
    """Emit a timestamped progress line immediately."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    thread_name = threading.current_thread().name
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)


def _make_orbit_camera(
    bot_red_com: np.ndarray,
    bot_blue_com: np.ndarray,
    ring_top_z: float,
    t: float,
    duration: float,
    base_azimuth: float = 90.0,
    elevation: float = -25.0,
    min_distance: float = 8.0,
    padding_factor: float = 1.8,
    orbit_range: float = 15.0,
) -> mujoco.MjvCamera:
    """Create a cinematic orbit camera that slowly rotates around the arena.

    Combines Sim2Reason-style smooth orbit with bot-tracking framing.

    Args:
        bot_red_com: Red bot center-of-mass position.
        bot_blue_com: Blue bot center-of-mass position.
        ring_top_z: Z height of ring surface.
        t: Current time in seconds.
        duration: Total match duration in seconds.
        base_azimuth: Center azimuth angle in degrees.
        elevation: Camera elevation angle in degrees.
        min_distance: Minimum camera distance.
        padding_factor: Multiply bot separation by this for camera distance.
        orbit_range: Total degrees swept during orbit (centered on base_azimuth).

    Returns:
        MjvCamera configured for orbit.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Lookat: midpoint between bots (smoothed externally for cinematic drift)
    midpoint = (bot_red_com + bot_blue_com) / 2.0
    cam.lookat[0] = float(midpoint[0])
    cam.lookat[1] = float(midpoint[1]) if midpoint.size >= 2 else 0.0
    cam.lookat[2] = ring_top_z

    # Distance: adaptive to bot separation (smoothed externally)
    separation = float(np.linalg.norm(bot_red_com - bot_blue_com))
    cam.distance = max(min_distance, separation * padding_factor)

    # Azimuth: smooth sinusoidal orbit
    progress = t / max(duration, 1.0)
    cam.azimuth = base_azimuth + orbit_range * 0.5 * np.cos(2.0 * np.pi * progress)
    cam.elevation = elevation

    return cam


def _make_tracking_camera(
    bot_red_com: np.ndarray,
    bot_blue_com: np.ndarray,
    ring_top_z: float,
    is_3d: bool,
    elevation: float = -30.0,
    azimuth: float = 90.0,
    min_distance: float = 6.0,
    padding_factor: float = 2.0,
) -> mujoco.MjvCamera:
    """Create a dynamic tracking camera that follows both bots.

    The camera looks at the midpoint between both COMs and zooms out
    far enough to keep both in frame.

    Args:
        bot_red_com: Red bot center-of-mass position.
        bot_blue_com: Blue bot center-of-mass position.
        ring_top_z: Z height of ring surface (camera lookat height).
        is_3d: Whether this is a 3D match (affects azimuth).
        elevation: Camera elevation angle in degrees.
        azimuth: Camera azimuth angle in degrees.
        min_distance: Minimum camera distance.
        padding_factor: Multiply bot separation by this for camera distance.

    Returns:
        MjvCamera configured for tracking.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    midpoint = (bot_red_com + bot_blue_com) / 2.0
    cam.lookat[0] = float(midpoint[0])
    # For 3D positions (shape 3), midpoint[1] is Y; for 2D (shape 2), it's the second coord
    cam.lookat[1] = float(midpoint[1]) if midpoint.size >= 2 else 0.0
    cam.lookat[2] = ring_top_z

    separation = float(np.linalg.norm(bot_red_com - bot_blue_com))
    cam.distance = max(min_distance, separation * padding_factor)
    cam.elevation = elevation
    cam.azimuth = azimuth

    return cam


def _build_obs_log_entry(
    t: int,
    red_obs: BotObservation,
    blue_obs: BotObservation,
    bot_red: BotRuntime,
    bot_blue: BotRuntime,
    blue_displacement: float = 0.0,
) -> dict:
    """Build a full observation log entry matching the frontend ObsLogEntry schema.

    Args:
        blue_displacement: Distance blue moved since previous step (meters).
    """
    return {
        "t": t,
        "max_t": int(red_obs.max_t),
        "red_pos": red_obs.my_pos.tolist(),
        "blue_pos": red_obs.opponent_pos.tolist(),
        "red_yaw": round(float(red_obs.my_yaw), 4),
        "red_pitch": round(float(red_obs.my_pitch), 4),
        "red_roll": round(float(red_obs.my_roll), 4),
        "blue_yaw": round(float(red_obs.opponent_yaw), 4),
        "blue_pitch": round(float(red_obs.opponent_pitch), 4),
        "blue_roll": round(float(red_obs.opponent_roll), 4),
        "red_velocity": [round(float(v), 4) for v in red_obs.my_velocity],
        "red_angular_velocity": [round(float(v), 4) for v in red_obs.my_angular_velocity],
        "blue_velocity": [round(float(v), 4) for v in red_obs.opponent_velocity],
        "blue_angular_velocity": [round(float(v), 4) for v in red_obs.opponent_angular_velocity],
        "dist_to_opponent": round(float(red_obs.distance_to_opponent), 4),
        "red_edge_dist": round(float(red_obs.my_edge_distance), 4),
        "blue_edge_dist": round(float(red_obs.opponent_edge_distance), 4),
        "contact": bool(red_obs.opponent_contact),
        "contact_force": round(float(red_obs.opponent_contact_force), 2),
        "red_ground_contact": bool(red_obs.ground_contact),
        "red_is_tipping": round(float(red_obs.is_tipping), 4),
        "blue_ground_contact": bool(blue_obs.ground_contact),
        "blue_is_tipping": round(float(blue_obs.is_tipping), 4),
        "red_mass": round(float(bot_red.total_mass), 3),
        "blue_mass": round(float(bot_blue.total_mass), 3),
        "red_bounding_radius": round(float(bot_red.bounding_radius), 3),
        "blue_bounding_radius": round(float(bot_blue.bounding_radius), 3),
        "ring_radius": round(float(red_obs.ring_radius), 3),
        "blue_displacement": round(blue_displacement, 4),
    }


def _build_action_log_entry(
    t: int,
    actions_red: np.ndarray,
    actions_blue: np.ndarray,
    bot_red: BotRuntime,
    bot_blue: BotRuntime,
) -> dict:
    """Build a full action log entry matching the frontend ActionLogEntry schema."""
    return {
        "t": t,
        "red_actions": {
            name: round(float(actions_red[i]), 3)
            for i, name in enumerate(bot_red.actuator_names)
        },
        "blue_actions": {
            name: round(float(actions_blue[i]), 3)
            for i, name in enumerate(bot_blue.actuator_names)
        },
    }


class Match:
    """Convenience wrapper to run two bot runtimes inside SumoEnv."""

    def __init__(
        self,
        env: SumoEnv,
        bot_red: BotRuntime,
        bot_blue: BotRuntime,
        max_steps: Optional[int] = None,
        headless: bool = False,
        seed: int = 0,
        score_function: str = "any",
        observation_config: Optional["ObservationConfig"] = None,
        match_time: Optional[float] = None,
        game_context: Optional[Dict[str, Any]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.bot_red = bot_red
        self.bot_blue = bot_blue
        self.max_steps = max_steps
        self.headless = headless
        self.seed = seed
        self.score_function = score_function
        self.is_3d = getattr(env, '_boundary_shape', '') == 'cylinder'
        self.game_context = dict(game_context or {})
        self.initial_state = dict(initial_state or {})
        setattr(self.env, "_stage2x_no_opponent", bool(self.game_context.get("__stage2x_no_opponent__")))

        # Resolve match_time → max_steps
        if match_time is not None:
            self.max_steps = int(match_time / self.env.control_timestep)

        # Observation grid config and cache (lazy import to avoid circular dep)
        if observation_config is None:
            from mjarena.core.build_config import ObservationConfig
            observation_config = ObservationConfig()
        self._obs_cfg = observation_config
        self._grid_cache: Optional[GridCache] = None
        ring_r = float(getattr(env, 'ring_radius', 5.0))
        needs_grids = (self._obs_cfg.arena_grid
                       or self._obs_cfg.arena_mass_grid
                       or self._obs_cfg.edge_distance_grid)
        if needs_grids:
            self._grid_cache = make_grid_cache(ring_r, self._obs_cfg.arena_grid_cell_size)

        from mjarena.envs.detailed_observations import DetailedObservations
        self.env.detailed_observations = DetailedObservations(self.env)

        # EMA smoothing state for tracking camera
        self._cam_smooth_lookat: Optional[np.ndarray] = None
        self._cam_smooth_distance: Optional[float] = None
        self._cam_smooth_alpha: float = 0.03  # lower = smoother

    def build_bot_observation(self, for_red: bool, time_step: int) -> BotObservation:
        """Build full 3D observation for a bot.

        Args:
            for_red: If True, build observation from red's perspective
            time_step: Current timestep
        """
        my_bot = self.bot_red if for_red else self.bot_blue
        opp_bot = self.bot_blue if for_red else self.bot_red
        my_prefix = my_bot.prefix
        no_opponent = bool(self.game_context.get("__stage2x_no_opponent__"))
        opp_prefix = opp_bot.prefix

        detail = self.env.detailed_observations.for_robot(my_prefix)
        detail["elapsed_time"] = time_step * detail["control_dt"]
        detail["time_remaining"] = max(0, self.max_steps - time_step) * detail["control_dt"]
        my_state, opp_state = detail["my_robot"], detail["opponent_robot"]
        my_root = my_state["bodies"][my_state["root_body"]]
        opp_root = opp_state["bodies"][opp_state["root_body"]]
        snapshot = self.env.detailed_observations.data
        my_matrix = np.asarray(snapshot.xmat[self.env._root_body_ids[my_prefix]]).reshape(3, 3)
        opp_matrix = np.asarray(snapshot.xmat[self.env._root_body_ids[opp_prefix]]).reshape(3, 3)
        my_position = my_state["com_position"]
        opp_position = opp_state["com_position"]

        my_edge = self.env.closest_distance_to_boundary(my_position)
        opp_edge = self.env.closest_distance_to_boundary(opp_position)

        # Contact info
        opp_contact, opp_force, gnd_contact = self.env.get_contact_info(my_prefix)

        my_pos_arr = np.array(my_position, dtype=np.float32)
        opp_pos_arr = np.array(opp_position, dtype=np.float32)
        my_br = float(my_bot.bounding_radius)
        opp_br = float(opp_bot.bounding_radius)
        my_mass = float(my_bot.total_mass)
        opp_mass = float(opp_bot.total_mass)
        ring_r = float(getattr(self.env, 'ring_radius', 5.0))

        if no_opponent:
            opp_pos_arr = np.array([1e6, 1e6, 0.0], dtype=np.float32)
            opp_edge = 1e6
            opp_contact = False
            opp_force = 0.0
            opp_br = 0.0
            opp_mass = 0.0

        # Build spatial grids if enabled
        obs_cfg = self._obs_cfg
        gc = self._grid_cache

        arena_grid = None
        arena_mass_grid = None
        edge_dist_grid = None

        if gc is not None:
            if obs_cfg.arena_grid:
                arena_grid = build_arena_grid(gc, my_pos_arr, opp_pos_arr, my_br, opp_br)
            if obs_cfg.arena_mass_grid:
                arena_mass_grid = build_arena_mass_grid(
                    gc, my_pos_arr, opp_pos_arr, my_br, opp_br, my_mass, opp_mass,
                )
            if obs_cfg.edge_distance_grid:
                edge_dist_grid = gc.edge_distance_grid

        return BotObservation(
            details=detail,
            # Position
            my_pos=my_pos_arr,
            opponent_pos=opp_pos_arr,
            # Orientation
            my_yaw=float(np.arctan2(my_matrix[1, 0], my_matrix[0, 0])),
            my_pitch=float(np.arcsin(np.clip(-my_matrix[2, 0], -1, 1))),
            my_roll=float(np.arctan2(my_matrix[2, 1], my_matrix[2, 2])),
            opponent_yaw=0.0 if no_opponent else float(np.arctan2(opp_matrix[1, 0], opp_matrix[0, 0])),
            opponent_pitch=0.0 if no_opponent else float(np.arcsin(np.clip(-opp_matrix[2, 0], -1, 1))),
            opponent_roll=0.0 if no_opponent else float(np.arctan2(opp_matrix[2, 1], opp_matrix[2, 2])),
            # Velocity
            my_velocity=np.array(my_root["linear_velocity"], dtype=np.float32),
            my_angular_velocity=np.array(my_matrix.T @ my_root["angular_velocity"], dtype=np.float32),
            opponent_velocity=(
                np.zeros(3, dtype=np.float32)
                if no_opponent
                else np.array(opp_root["linear_velocity"], dtype=np.float32)
            ),
            opponent_angular_velocity=(
                np.zeros(3, dtype=np.float32)
                if no_opponent
                else np.array(opp_matrix.T @ opp_root["angular_velocity"], dtype=np.float32)
            ),
            my_actuator_velocity=my_bot.actuator_joint_velocity(snapshot),
            opponent_actuator_velocity=(
                np.zeros(0, dtype=np.float32)
                if no_opponent
                else opp_bot.actuator_joint_velocity(snapshot)
            ),
            # Distances
            distance_to_opponent=(
                1e6 if no_opponent else float(np.linalg.norm(my_position - opp_position))
            ),
            my_edge_distance=float(my_edge),
            opponent_edge_distance=float(opp_edge),
            # Contact
            opponent_contact=bool(opp_contact),
            opponent_contact_force=float(opp_force),
            ground_contact=bool(gnd_contact),
            # Stability
            is_tipping=float(np.clip(np.arccos(np.clip(my_matrix[2, 2], -1, 1)) / (np.pi / 2), 0, 1)),
            # Time
            t=int(time_step),
            max_t=int(self.max_steps),
            game={
                str(k): v
                for k, v in self.game_context.items()
                if not str(k).startswith("__")
            },
            # Robot properties
            my_bounding_radius=my_br,
            opponent_bounding_radius=opp_br,
            ring_radius=ring_r,
            # Spatial grids
            arena_grid=arena_grid,
            arena_mass_grid=arena_mass_grid,
            edge_distance_grid=edge_dist_grid,
            # Inactivity timers
            my_inactivity_timer=getattr(self.env, '_inactivity_timers', {"red": 0.0, "blue": 0.0}).get("red" if for_red else "blue", 0.0),
            opponent_inactivity_timer=getattr(self.env, '_inactivity_timers', {"red": 0.0, "blue": 0.0}).get("blue" if for_red else "red", 0.0),
        )
        
    def single_match_step(
        self,
        match_step: int,
        verbose: bool = True,
        *,
        trace_label: Optional[str] = None,
        trace_step: bool = False,
    ):
            step_t0 = time.monotonic()

            # --- build observations for each agent ---
            if trace_step:
                _trace(f"{trace_label or 'match'}: step {match_step} build observations start", enabled=True)
            obs_t0 = time.monotonic()
            red_obs = self.build_bot_observation(for_red=True, time_step=self.env.t)
            blue_obs = self.build_bot_observation(for_red=False, time_step=self.env.t)
            obs_elapsed = time.monotonic() - obs_t0
            if trace_step:
                _trace(
                    f"{trace_label or 'match'}: step {match_step} build observations done in {obs_elapsed:.3f}s",
                    enabled=True,
                )

            # --- query both agents for actions ---
            if trace_step:
                _trace(f"{trace_label or 'match'}: step {match_step} controller act start", enabled=True)
            act_t0 = time.monotonic()
            # Forfeit-on-crash: if a controller raises while choosing an action,
            # that bot forfeits the match (the opponent wins). We substitute a
            # no-op (zero) action so the physics tick still completes cleanly,
            # and flag the crashing color via self._forfeit_color; the match loop
            # reads it and ends the match with termination_reason="forfeit_crash".
            red_crashed = blue_crashed = False
            try:
                actions_red = self.bot_red.act(red_obs)
            except Exception as exc:
                red_crashed = True
                actions_red = np.zeros(len(self.bot_red.actuator_names))
                if "red" not in self._controller_errors_by_color:
                    self._controller_errors_by_color["red"] = {
                        "color": "red",
                        "step": int(match_step),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                if not self._forfeit_logged:
                    logger.warning("red controller crashed at step %d (forfeit): %s", match_step, exc)
            try:
                actions_blue = self.bot_blue.act(blue_obs)
            except Exception as exc:
                blue_crashed = True
                actions_blue = np.zeros(len(self.bot_blue.actuator_names))
                if "blue" not in self._controller_errors_by_color:
                    self._controller_errors_by_color["blue"] = {
                        "color": "blue",
                        "step": int(match_step),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                if not self._forfeit_logged:
                    logger.warning("blue controller crashed at step %d (forfeit): %s", match_step, exc)
            if red_crashed or blue_crashed:
                self._forfeit_logged = True
                self._forfeit_color = "both" if (red_crashed and blue_crashed) else ("red" if red_crashed else "blue")
            else:
                self._forfeit_color = None
            act_elapsed = time.monotonic() - act_t0
            if trace_step:
                _trace(
                    f"{trace_label or 'match'}: step {match_step} controller act done in {act_elapsed:.3f}s",
                    enabled=True,
                )

            # --- apply one physics tick ---
            if trace_step:
                _trace(f"{trace_label or 'match'}: step {match_step} env.step start", enabled=True)
            env_t0 = time.monotonic()
            _, _, terminated, truncated, info = self.env.step({"red": actions_red, "blue": actions_blue})
            env_elapsed = time.monotonic() - env_t0
            if trace_step:
                _trace(
                    f"{trace_label or 'match'}: step {match_step} env.step done in {env_elapsed:.3f}s "
                    f"(total={time.monotonic() - step_t0:.3f}s)",
                    enabled=True,
                )

            return red_obs, blue_obs, actions_red, actions_blue, terminated, truncated, info

    def _render_frame(self, mjrenderer, writer, camera_mode, overlay_info, post_win=False):
        """Render one frame, supporting fixed, tracking, and orbit camera modes."""
        if camera_mode in ("tracking", "orbit"):
            if camera_mode == "orbit":
                sim_time = float(self.env.data.time)
                duration = float(self.max_steps * self.env.model.opt.timestep)
                cam = _make_orbit_camera(
                    self.bot_red.com_position,
                    self.bot_blue.com_position,
                    self.env.ring_top_z,
                    t=sim_time,
                    duration=duration,
                )
            else:
                cam = _make_tracking_camera(
                    self.bot_red.com_position,
                    self.bot_blue.com_position,
                    self.env.ring_top_z,
                    self.is_3d,
                )

            # EMA smoothing
            if camera_mode == "tracking":
                # Smooth both lookat and distance
                raw_lookat = np.array([cam.lookat[0], cam.lookat[1], cam.lookat[2]])
                raw_distance = cam.distance
                alpha = self._cam_smooth_alpha

                if self._cam_smooth_lookat is None:
                    self._cam_smooth_lookat = raw_lookat
                    self._cam_smooth_distance = raw_distance
                else:
                    self._cam_smooth_lookat = alpha * raw_lookat + (1 - alpha) * self._cam_smooth_lookat
                    self._cam_smooth_distance = alpha * raw_distance + (1 - alpha) * self._cam_smooth_distance

                cam.lookat[0] = float(self._cam_smooth_lookat[0])
                cam.lookat[1] = float(self._cam_smooth_lookat[1])
                cam.lookat[2] = float(self._cam_smooth_lookat[2])
                cam.distance = self._cam_smooth_distance
            elif camera_mode == "orbit":
                # Heavy smoothing on both lookat and distance for cinematic drift
                raw_lookat = np.array([cam.lookat[0], cam.lookat[1], cam.lookat[2]])
                raw_distance = cam.distance
                orbit_alpha = 0.01  # very heavy — slow cinematic drift

                if self._cam_smooth_lookat is None:
                    self._cam_smooth_lookat = raw_lookat
                    self._cam_smooth_distance = raw_distance
                else:
                    self._cam_smooth_lookat = orbit_alpha * raw_lookat + (1 - orbit_alpha) * self._cam_smooth_lookat
                    self._cam_smooth_distance = orbit_alpha * raw_distance + (1 - orbit_alpha) * self._cam_smooth_distance

                cam.lookat[0] = float(self._cam_smooth_lookat[0])
                cam.lookat[1] = float(self._cam_smooth_lookat[1])
                cam.lookat[2] = float(self._cam_smooth_lookat[2])
                cam.distance = self._cam_smooth_distance

            # After a win, clamp camera near arena center so surviving robot stays visible
            if post_win and camera_mode == "tracking":
                xy = np.array([cam.lookat[0], cam.lookat[1]])
                dist = np.linalg.norm(xy)
                max_r = getattr(self.env, 'ring_radius', 3.0) * 0.5
                if dist > max_r:
                    scale = max_r / dist
                    cam.lookat[0] = float(xy[0] * scale)
                    cam.lookat[1] = float(xy[1] * scale)
                cam.distance = max(cam.distance, 8.0)

            mjrenderer.update_scene(self.env.data, camera=cam, scene_option=self._vopt)
        else:
            mjrenderer.update_scene(self.env.data, camera=camera_mode, scene_option=self._vopt)
        rgb = mjrenderer.render()
        capture_frame(writer, rgb, overlay_info=overlay_info)

    def _apply_initial_state(self) -> None:
        """Apply optional per-match initial state after env.reset settling."""
        if not self.initial_state:
            return
        if not isinstance(self.initial_state, Mapping):
            raise ValueError("initial_state must be an object.")
        spawn_joints = getattr(self.env, "_spawn_joints", None)
        if not spawn_joints:
            raise ValueError("initial_state requested, but robot spawn joints are unavailable.")

        for actor in ("red", "blue"):
            spec = self.initial_state.get(actor)
            if spec is None:
                continue
            if not isinstance(spec, Mapping):
                raise ValueError(f"initial_state.{actor} must be an object.")
            self._apply_actor_initial_position(actor, spec, spawn_joints)

        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        for actor in ("red", "blue"):
            spec = self.initial_state.get(actor)
            if isinstance(spec, Mapping):
                self._apply_actor_initial_yaw(actor, spec, spawn_joints)

        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        settle_steps = int(self.initial_state.get("settle_steps", 0) or 0)
        if settle_steps < 0 or settle_steps > 500:
            raise ValueError("initial_state.settle_steps must be between 0 and 500.")
        for _ in range(settle_steps):
            self.env.data.ctrl[:] = 0.0
            mujoco.mj_step(self.env.model, self.env.data)
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def _apply_actor_initial_position(self, actor: str, spec: Mapping[str, Any], spawn_joints: Mapping[str, Any]) -> None:
        pos = spec.get("pos", spec.get("position"))
        if pos is None:
            return
        if not isinstance(pos, (list, tuple)) or len(pos) not in (2, 3):
            raise ValueError(f"initial_state.{actor}.pos must be [x, y] or [x, y, z].")
        joint_info = spawn_joints.get(actor)
        if not joint_info:
            raise ValueError(f"initial_state.{actor} requested, but {actor} spawn joint is unavailable.")
        joint_type, qpos_adr, base = self._initial_state_joint_parts(joint_info)
        if joint_type == "free":
            self.env.data.qpos[qpos_adr + 0] = float(pos[0])
            self.env.data.qpos[qpos_adr + 1] = float(pos[1])
            if len(pos) == 3:
                self.env.data.qpos[qpos_adr + 2] = self._resolve_initial_state_z(float(pos[2]))
            return

        if len(pos) > 1 and abs(float(pos[1])) > 1e-6:
            raise ValueError(f"initial_state.{actor}.pos y-coordinate requires a freejoint/3D robot.")
        self.env.data.qpos[qpos_adr] = float(pos[0]) - float(base)

    def _apply_actor_initial_yaw(self, actor: str, spec: Mapping[str, Any], spawn_joints: Mapping[str, Any]) -> None:
        yaw = self._resolve_initial_yaw(actor, spec)
        if yaw is None:
            return
        joint_info = spawn_joints.get(actor)
        if not joint_info:
            raise ValueError(f"initial_state.{actor} requested, but {actor} spawn joint is unavailable.")
        joint_type, qpos_adr, _base = self._initial_state_joint_parts(joint_info)
        if joint_type != "free":
            raise ValueError(f"initial_state.{actor}.yaw requires a freejoint/3D robot.")
        self.env.data.qpos[qpos_adr + 3] = math.cos(yaw / 2.0)
        self.env.data.qpos[qpos_adr + 4] = 0.0
        self.env.data.qpos[qpos_adr + 5] = 0.0
        self.env.data.qpos[qpos_adr + 6] = math.sin(yaw / 2.0)

    def _resolve_initial_yaw(self, actor: str, spec: Mapping[str, Any]) -> Optional[float]:
        if "yaw" in spec:
            return float(spec["yaw"])
        if "yaw_degrees" in spec:
            return math.radians(float(spec["yaw_degrees"]))

        target = spec.get("yaw_toward")
        face = spec.get("face")
        if target is None and isinstance(face, str):
            if face == "center":
                target = [0.0, 0.0]
            elif face == "opponent":
                other = "blue" if actor == "red" else "red"
                target = self._actor_xy(other)
            elif face:
                raise ValueError(f"initial_state.{actor}.face must be 'center' or 'opponent'.")
        if target is None:
            return None
        if not isinstance(target, (list, tuple, np.ndarray)) or len(target) < 2:
            raise ValueError(f"initial_state.{actor}.yaw_toward must be [x, y].")
        pos = self._actor_xy(actor)
        return math.atan2(float(target[1]) - pos[1], float(target[0]) - pos[0])

    def _actor_xy(self, actor: str) -> tuple[float, float]:
        bot = self.bot_red if actor == "red" else self.bot_blue
        pos = bot.com_position
        return float(pos[0]), float(pos[1])

    def _resolve_initial_state_z(self, z: float) -> float:
        """Resolve Stage 2X local z heights into world z coordinates."""
        if not self.game_context.get("__initial_state_z_relative_to_ring__"):
            return z
        ring_top = float(getattr(self.env, "ring_top_z", 0.0) or 0.0)
        if z < ring_top - 0.1:
            return ring_top + z
        return z

    @staticmethod
    def _initial_state_joint_parts(joint_info: Any) -> tuple[str, int, Any]:
        if len(joint_info) == 4:
            _joint_id, qpos_adr, base, joint_type = joint_info
            return str(joint_type), int(qpos_adr), base
        _joint_id, qpos_adr, base = joint_info
        return "slide", int(qpos_adr), base

    def run(self, verbose: bool = False, save_video: bool = True,
            video_path: Optional[str] = None,
            video_width: int = 640, video_height: int = 480,
            video_fps: Optional[int] = None, camera_mode: str = "topdown",
            quiet: bool = False,
            overlay_info: Optional[VideoOverlayInfo] = None,
            trace_label: Optional[str] = None,
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            rendering_flags: Optional[Dict[str, bool]] = None,
            ) -> GameRecord:
        """Run a match.

        Args:
            quiet: If True, suppress all logging output.
        """
        trace_enabled = bool(trace_label)
        heartbeat_every = max(100, int((self.max_steps or 1000) / 10))
        _trace(
            f"{trace_label or 'match'}: Match.run start "
            f"(max_steps={self.max_steps}, save_video={save_video}, camera={camera_mode})",
            enabled=trace_enabled,
        )

        match_step = 0
        winner: Optional[str] = None
        winner_step: Optional[int] = None
        inactivity_loser: Optional[str] = None
        termination_reason: str = "timeout"
        # Forfeit-on-crash state (set by single_match_step when a controller raises)
        self._forfeit_color: Optional[str] = None
        self._forfeit_logged: bool = False
        # First controller exception per side; later ones never replace it.
        self._controller_errors_by_color: Dict[str, Dict[str, Any]] = {}
        post_win_steps = 0
        POST_WIN_MAX_STEPS_GUI = 10000
        POST_WIN_MAX_STEPS_HEADLESS = 120
        post_win_limit = POST_WIN_MAX_STEPS_HEADLESS if self.headless else POST_WIN_MAX_STEPS_GUI
        progress_t0 = time.monotonic()
        last_progress_emit = 0.0
        progress_every = max(25, int((self.max_steps or 1000) / 20))

        # Columnar collectors for GameRecord timeseries
        red_positions_log = []
        blue_positions_log = []
        red_orientations_log = []
        blue_orientations_log = []
        red_velocities_log = []
        blue_velocities_log = []
        red_angular_velocities_log = []
        blue_angular_velocities_log = []
        distances_to_opponent_log = []
        red_edge_distances_log = []
        blue_edge_distances_log = []
        contacts_log = []
        contact_forces_log = []
        red_tipping_log = []
        blue_tipping_log = []
        red_ground_contacts_log = []
        blue_ground_contacts_log = []
        blue_displacements_log = []
        red_actions_log = []
        blue_actions_log = []
        red_inactivity_timers_log = []
        blue_inactivity_timers_log = []
        surface_gap_log = []
        qpos_log = []
        prev_blue_pos: Optional[np.ndarray] = None  # for blue_displacement tracking

        def _collect_columnar(red_obs, blue_obs, actions_red, actions_blue, blue_disp):
            """Append one timestep to all columnar collectors."""
            surface_gap_log.append(float(red_obs.details["opponent_surface_distance"]))
            red_positions_log.append(red_obs.my_pos.tolist())
            blue_positions_log.append(red_obs.opponent_pos.tolist())
            red_orientations_log.append({
                "yaw": round(float(red_obs.my_yaw), 4),
                "pitch": round(float(red_obs.my_pitch), 4),
                "roll": round(float(red_obs.my_roll), 4),
            })
            blue_orientations_log.append({
                "yaw": round(float(red_obs.opponent_yaw), 4),
                "pitch": round(float(red_obs.opponent_pitch), 4),
                "roll": round(float(red_obs.opponent_roll), 4),
            })
            red_velocities_log.append([round(float(v), 4) for v in red_obs.my_velocity])
            blue_velocities_log.append([round(float(v), 4) for v in red_obs.opponent_velocity])
            red_angular_velocities_log.append([round(float(v), 4) for v in red_obs.my_angular_velocity])
            blue_angular_velocities_log.append([round(float(v), 4) for v in red_obs.opponent_angular_velocity])
            distances_to_opponent_log.append(round(float(red_obs.distance_to_opponent), 4))
            red_edge_distances_log.append(round(float(red_obs.my_edge_distance), 4))
            blue_edge_distances_log.append(round(float(red_obs.opponent_edge_distance), 4))
            contacts_log.append(bool(red_obs.opponent_contact))
            contact_forces_log.append(round(float(red_obs.opponent_contact_force), 2))
            red_tipping_log.append(round(float(red_obs.is_tipping), 4))
            blue_tipping_log.append(round(float(blue_obs.is_tipping), 4))
            red_ground_contacts_log.append(bool(red_obs.ground_contact))
            blue_ground_contacts_log.append(bool(blue_obs.ground_contact))
            blue_displacements_log.append(round(blue_disp, 4))
            # Actions
            red_act = {
                name: round(float(actions_red[i]), 3)
                for i, name in enumerate(self.bot_red.actuator_names)
            }
            blue_act = {
                name: round(float(actions_blue[i]), 3)
                for i, name in enumerate(self.bot_blue.actuator_names)
            }
            red_actions_log.append(red_act)
            blue_actions_log.append(blue_act)
            # Inactivity timers
            inactivity_timers = getattr(self.env, '_inactivity_timers', {"red": 0.0, "blue": 0.0})
            red_inactivity_timers_log.append(inactivity_timers.get("red", 0.0))
            blue_inactivity_timers_log.append(inactivity_timers.get("blue", 0.0))
            # Full joint state for Blender re-rendering
            qpos_log.append([round(float(q), 6) for q in self.env.data.qpos])

        # Video setup (only for fixed-length runs; skip when open-ended viewer is used)
        writer = None
        mjrenderer = None
        actual_video_fps = video_fps
        if save_video:
            _trace(f"{trace_label or 'match'}: video setup start", enabled=trace_enabled)
            mjrenderer = mujoco.Renderer(self.env.model, video_height, video_width)
            self._vopt = mujoco.MjvOption()
            if rendering_flags is not None:
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = rendering_flags["show_joint"]
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = rendering_flags["show_autoconnect"]
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = rendering_flags["show_com"]
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = rendering_flags["show_contact_point"]
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = rendering_flags["show_contact_force"]
            else:
                self._vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
            writer = get_offscreen_writer(video_path, video_fps, self.env.control_timestep)
            # Resolve actual fps used by the writer
            if actual_video_fps is None:
                actual_video_fps = int(round(1.0 / float(self.env.control_timestep))) if self.env.control_timestep else 60
            _trace(
                f"{trace_label or 'match'}: video setup done (fps={actual_video_fps}, path={video_path})",
                enabled=trace_enabled,
            )
       
        def _is_match_done(terminated: bool, truncated: bool, info: Dict[str, Any]) -> str | None:
            # Runtime size limit — outranks inactivity and instability
            if info.get("termination_reason") == "size_violation":
                loser = info["size_violation_loser"]
                return True, {"red": "blue", "blue": "red", "both": "tie"}[loser]

            # Inactivity termination (from SumoEnv.step())
            if info.get("termination_reason") == "inactivity":
                loser = info["inactivity_loser"]
                if loser == "both":
                    return True, "tie"
                elif loser == "red":
                    return True, "blue"
                else:
                    return True, "red"

            # QACC instability — blame the bot that owns the unstable DOF
            if info.get("termination_reason") == "qacc":
                loser = info.get("qacc_loser", "both")
                if loser == "both":
                    return True, "tie"
                elif loser == "red":
                    return True, "blue"
                else:
                    return True, "red"

            if self.score_function == "thres-50":
                # >50% of robot mass touching outside floor = loss.
                # Allows projectiles: small pieces can fly off without self-losing.
                red_over = info.get("red_floor_mass_frac", 0.0) > 0.5
                blue_over = info.get("blue_floor_mass_frac", 0.0) > 0.5
                if truncated:
                    return True, "tie"
                if red_over and not blue_over:
                    return True, "blue"
                elif blue_over and not red_over:
                    return True, "red"
                elif red_over and blue_over:
                    return True, "tie"
                return False, None
            else:
                # "any" — current default: any floor contact = instant loss
                red_hit_floor = info.get("red_hit_floor", False)
                blue_hit_floor = info.get("blue_hit_floor", False)

                if terminated or truncated:
                    done, winner = True, "tie"
                else:
                    done, winner = False, None

                if red_hit_floor and not blue_hit_floor:
                    done, winner = True, "blue"
                elif blue_hit_floor and not red_hit_floor:
                    done, winner = True, "red"

                return done, winner

        def _emit_progress(step: int, *, done: bool = False, winner: Optional[str] = None) -> None:
            nonlocal last_progress_emit
            if progress_callback is None:
                return
            now = time.monotonic()
            if not done and step != 0 and (now - last_progress_emit) < 1.0 and step % progress_every != 0:
                return
            elapsed_sec = now - progress_t0
            max_steps = self.max_steps
            steps_left = None
            eta_sec = None
            if max_steps is not None:
                steps_left = max(max_steps - step, 0)
                if step > 0 and elapsed_sec > 0:
                    eta_sec = (elapsed_sec / step) * steps_left
            progress_callback({
                "type": "done" if done else "progress",
                "step": step,
                "max_steps": max_steps,
                "steps_left": steps_left,
                "eta_sec": eta_sec,
                "elapsed_sec": elapsed_sec,
                "winner": winner,
            })
            last_progress_emit = now
        
        # Interactive loop: runs until you close the window or a robot leaves the ring
        _trace(f"{trace_label or 'match'}: env.reset start (seed={self.seed})", enabled=trace_enabled)
        self.env.reset(seed=self.seed)
        self._apply_initial_state()
        self.env.maybe_sync_viewer()
        _trace(f"{trace_label or 'match'}: env.reset done", enabled=trace_enabled)
        _emit_progress(0)
        initial_qpos = [float(q) for q in self.env.data.qpos]

        # Capture initial COM positions for progress tracking
        initial_red_com = self.bot_red.com_position.copy()
        initial_blue_com = self.bot_blue.com_position.copy()
        initial_distance = float(np.linalg.norm(initial_red_com - initial_blue_com))

        # Capture an initial frame to avoid a black first frame in headless/video
        if writer is not None and mjrenderer is not None:
            _trace(f"{trace_label or 'match'}: initial render start", enabled=trace_enabled)
            self._render_frame(mjrenderer, writer, camera_mode, overlay_info)
            _trace(f"{trace_label or 'match'}: initial render done", enabled=trace_enabled)
        
        # set tqdm display (Gladiator Arena 1v1)
        tqdm_kwargs = {"desc": "Gladiator Arena 1v1", "leave": True} if verbose else {"disable": quiet}

        try:
            # if GUI mode, run until the window is closed
            if not self.headless:
                while self.env.viewer.is_running():
                    # --- single match step ---
                    if winner is None:
                        red_obs, blue_obs, actions_red, actions_blue, terminated, truncated, info = self.single_match_step(match_step, verbose)

                        # Collect columnar data
                        cur_blue = red_obs.opponent_pos.copy()
                        blue_disp = float(np.linalg.norm(cur_blue - prev_blue_pos)) if prev_blue_pos is not None else 0.0
                        prev_blue_pos = cur_blue
                        _collect_columnar(red_obs, blue_obs, actions_red, actions_blue, blue_disp)

                        # --- sync the GUI: redraw + ingest user inputs ---
                        self.env.maybe_sync_viewer()
                        time.sleep(0.01)

                        # --- update video ---
                        if writer is not None:
                            self._render_frame(mjrenderer, writer, camera_mode, overlay_info)

                        # --- get winner ---
                        done, new_winner = _is_match_done(terminated, truncated, info)
                        if self._forfeit_color is not None:
                            done = True
                            new_winner = {"red": "blue", "blue": "red", "both": "tie"}[self._forfeit_color]
                            info["termination_reason"] = "forfeit_crash"
                        if done:
                            winner = new_winner
                            winner_step = match_step
                            post_win_steps = 0
                            inactivity_loser = info.get("inactivity_loser")
                            termination_reason = info.get("termination_reason") or "timeout"
                            self.env.set_winner_lighting(winner)
                            if overlay_info is not None:
                                overlay_info.winner = winner
                            if inactivity_loser and not quiet:
                                logger.info(f"Inactivity timeout: {inactivity_loser} bot(s) inactive")
                        match_step += 1
                    else:
                        continue

                self.env.maybe_close_viewer()
            else:
                _trace(f"{trace_label or 'match'}: entering headless loop", enabled=trace_enabled)
                with tqdm(range(self.max_steps), **tqdm_kwargs) as progress_bar:
                    done = False
                    winner = None
                    for _ in progress_bar:
                        if not done:
                            trace_this_step = trace_enabled and (match_step == 0 or match_step % heartbeat_every == 0)
                            if trace_this_step:
                                _trace(
                                    f"{trace_label or 'match'}: loop step {match_step} start",
                                    enabled=True,
                                )
                            red_obs, blue_obs, actions_red, actions_blue, terminated, truncated, info = self.single_match_step(
                                match_step,
                                verbose,
                                trace_label=trace_label,
                                trace_step=trace_this_step,
                            )

                            # Collect columnar data
                            cur_blue = red_obs.opponent_pos.copy()
                            blue_disp = float(np.linalg.norm(cur_blue - prev_blue_pos)) if prev_blue_pos is not None else 0.0
                            prev_blue_pos = cur_blue
                            _collect_columnar(red_obs, blue_obs, actions_red, actions_blue, blue_disp)

                            if writer is not None:
                                if trace_this_step:
                                    _trace(
                                        f"{trace_label or 'match'}: step {match_step} render start",
                                        enabled=True,
                                    )
                                render_t0 = time.monotonic()
                                self._render_frame(mjrenderer, writer, camera_mode, overlay_info)
                                if trace_this_step:
                                    _trace(
                                        f"{trace_label or 'match'}: step {match_step} render done in {time.monotonic() - render_t0:.3f}s",
                                        enabled=True,
                                    )

                            done, winner = _is_match_done(terminated, truncated, info)
                            if self._forfeit_color is not None:
                                done = True
                                winner = {"red": "blue", "blue": "red", "both": "tie"}[self._forfeit_color]
                                info["termination_reason"] = "forfeit_crash"
                            winner_step = match_step
                            if trace_this_step:
                                _trace(
                                    f"{trace_label or 'match'}: step {match_step} outcome check done "
                                    f"(done={done}, winner={winner})",
                                    enabled=True,
                                )

                            if done:
                                inactivity_loser = info.get("inactivity_loser")
                                termination_reason = info.get("termination_reason") or "timeout"
                                if inactivity_loser and not quiet:
                                    logger.info(f"Inactivity timeout: {inactivity_loser} bot(s) inactive")
                                _trace(
                                    f"{trace_label or 'match'}: match resolved at step {match_step} "
                                    f"(winner={winner}, termination={termination_reason})",
                                    enabled=trace_enabled,
                                )

                            progress_bar.set_postfix_str(
                                format_tqdm_bar(match_step, info.get("distance_to_opponent"), winner),
                                refresh=False,
                            )
                            if trace_enabled and match_step > 0 and match_step % heartbeat_every == 0:
                                _trace(
                                    f"{trace_label or 'match'}: heartbeat step {match_step}/{self.max_steps} "
                                    f"(done={done}, winner={winner})",
                                    enabled=True,
                                )
                            executed_steps = match_step + 1
                            if done:
                                _emit_progress(executed_steps, winner=winner)
                                self.env.set_winner_lighting(winner)
                                if overlay_info is not None:
                                    overlay_info.winner = winner
                                break
                            _emit_progress(executed_steps)
                            match_step += 1
                    progress_bar.refresh()
                    if not done:
                        _emit_progress(self.max_steps or match_step, done=True, winner=winner or "tie")

                    # winner is found, we just run for an extra post_win steps to make the video nicer
                    # this simulates the thing falling off screen
                    if not (winner == "tie"):
                        _trace(
                            f"{trace_label or 'match'}: post-win frames start ({POST_WIN_MAX_STEPS_HEADLESS} steps)",
                            enabled=trace_enabled,
                        )
                        for _ in range(POST_WIN_MAX_STEPS_HEADLESS):
                            _, _, _, _, terminated, truncated, info = self.single_match_step(match_step, verbose)
                            if writer is not None:
                                self._render_frame(mjrenderer, writer, camera_mode, overlay_info, post_win=True)
                        _trace(f"{trace_label or 'match'}: post-win frames done", enabled=trace_enabled)
                    if done:
                        resolved_steps = (winner_step + 1) if winner_step is not None else (self.max_steps or match_step)
                        _emit_progress(resolved_steps, done=True, winner=winner or "tie")

        finally:
            if mjrenderer is not None:
                mjrenderer.close()
            if writer is not None:
                try:
                    _trace(f"{trace_label or 'match'}: writer.close start", enabled=trace_enabled)
                    writer.close()
                    _trace(f"{trace_label or 'match'}: writer.close done", enabled=trace_enabled)
                    if not quiet:
                        logging.getLogger().info(f"Video saved: {video_path} (at step {match_step})")
                except Exception as e:
                    raise Exception(f"Failed to save video to: {video_path}")

        if not quiet:
            logging.getLogger().info("="*50)
            display_winner = winner or "tie"
            resolved_step = winner_step if winner_step is not None else match_step
            logging.getLogger().info(f"Match ended with winner: {display_winner} at step {resolved_step}")
            if winner_step is not None and match_step > winner_step:
                extra_steps = match_step - winner_step
                if self.headless:
                    logging.getLogger().debug(f"Continued headless for {extra_steps} additional frames after the finish.")
                else:
                    logging.getLogger().debug(f"Continued for {extra_steps} additional GUI frames after the finish.")
            logging.getLogger().info("="*50)

        # Calculate final positions for progress
        final_red_com = self.bot_red.com_position.copy()
        final_blue_com = self.bot_blue.com_position.copy()
        final_distance = float(np.linalg.norm(final_red_com - final_blue_com))

        # Compute progress (fraction of initial distance closed)
        progress = 0.0
        if initial_distance > 0:
            distance_closed = initial_distance - final_distance
            progress = max(0.0, min(1.0, distance_closed / initial_distance))

        # Build GameRecord directly
        from mjarena.design_shop.types import GameRecord
        from mjarena.eval.metrics import compute_combat_score

        game_record = GameRecord(
            seed=self.seed,
            winner=winner or "tie",
            num_steps=len(qpos_log),
            winner_step=winner_step,
            progress=progress,
            ring_radius=float(self.env.ring_radius),
            red_mass=float(self.bot_red.total_mass),
            blue_mass=float(self.bot_blue.total_mass),
            red_bounding_radius=float(self.bot_red.bounding_radius),
            blue_bounding_radius=float(self.bot_blue.bounding_radius),
            red_actuator_names=[
                n[len(self.bot_red.prefix):] if self.bot_red.prefix else n
                for n in self.bot_red.actuator_names
            ],
            blue_actuator_names=[
                n[len(self.bot_blue.prefix):] if self.bot_blue.prefix else n
                for n in self.bot_blue.actuator_names
            ],
            red_positions=red_positions_log,
            blue_positions=blue_positions_log,
            red_orientations=red_orientations_log,
            blue_orientations=blue_orientations_log,
            red_velocities=red_velocities_log,
            blue_velocities=blue_velocities_log,
            red_angular_velocities=red_angular_velocities_log,
            blue_angular_velocities=blue_angular_velocities_log,
            distances_to_opponent=distances_to_opponent_log,
            red_edge_distances=red_edge_distances_log,
            blue_edge_distances=blue_edge_distances_log,
            contacts=contacts_log,
            contact_forces=contact_forces_log,
            red_tipping=red_tipping_log,
            blue_tipping=blue_tipping_log,
            red_ground_contacts=red_ground_contacts_log,
            blue_ground_contacts=blue_ground_contacts_log,
            blue_displacements=blue_displacements_log,
            red_actions=red_actions_log,
            blue_actions=blue_actions_log,
            physics_unstable=self.env.physics_unstable,
            qacc_warning_steps=list(self.env._qacc_warning_steps),
            qacc_gear_diagnostics=list(self.env._qacc_gear_diagnostics),
            qacc_loser=self.env._qacc_loser,
            qacc_dof_index=self.env._qacc_dof_index,
            qacc_body_name=self.env._qacc_body_name,
            red_inactivity_timers=red_inactivity_timers_log,
            blue_inactivity_timers=blue_inactivity_timers_log,
            red_surface_distances=surface_gap_log,
            qpos=qpos_log,
            initial_qpos=initial_qpos,
            control_dt=self.env.detailed_observations.control_dt,
            initial_red_pos=initial_red_com.tolist(),
            initial_blue_pos=initial_blue_com.tolist(),
            initial_distance=initial_distance,
            termination_reason=termination_reason if winner else "draw",
            controller_errors=(
                list(self._controller_errors_by_color.values())
                if termination_reason == "forfeit_crash"
                else []
            ),
        )

        # Compute combat metrics
        combat = compute_combat_score(game_record)
        game_record.combat_metrics = {
            "engagement": combat.engagement,
            "dominant_contact": combat.dominant_contact,
            "displacement": combat.displacement,
            "destabilization": combat.destabilization,
            "self_stability": combat.self_stability,
            "composite": combat.composite,
        }

        # Store video_fps for callers that need it
        self._last_video_fps = actual_video_fps
        _trace(
            f"{trace_label or 'match'}: Match.run done (winner={winner or 'tie'}, steps={match_step})",
            enabled=trace_enabled,
        )

        return game_record


def run_match(
    composed_xml: Path,
    red_policy_py: Callable[[BotObservation], Mapping[str, float]],
    blue_policy_py: Callable[[BotObservation], Mapping[str, float]],
    out_dir: Path,
    max_steps: Optional[int],
    use_gui: bool,
    save_video: bool,
    camera_mode: str,
    action_history_len: int = 10,
    env_cls: Type[SumoEnv] = SumoEnv,
    env_kwargs: Optional[Dict[str, Any]] = None,
    video_path: Optional[str] = None,
    seed: int = 0,
    quiet: bool = False,
    video_width: int = 640,
    video_height: int = 480,
    overlay_info: Optional[VideoOverlayInfo] = None,
    score_function: str = "any",
    max_obs_lookback: int = 20,
    max_action_lookback: int = 20,
    gear_clamp_ratio: float = 0.0,
    observation_config: Optional["ObservationConfig"] = None,
    match_time: Optional[float] = None,
    inactivity_timeout_seconds: Optional[float] = 10.0,
    inactivity_min_displacement: float = 0.5,
    inactivity_exempt_prefixes: Optional[list[str]] = None,
    size_limits: Optional[Tuple[float, float, float]] = None,
    trace_label: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    rendering_flags: Optional[Dict[str, bool]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> GameRecord:
    trace_enabled = bool(trace_label)
    _trace(f"{trace_label or 'match'}: _mj_load start", enabled=trace_enabled)

    # Bridge entrants' controllers to runtime bots.
    model, data = _mj_load(composed_xml)
    _trace(f"{trace_label or 'match'}: _mj_load done", enabled=trace_enabled)

    # Clamp actuator gears to prevent simulation instability (0 = disabled)
    if gear_clamp_ratio > 0:
        clamp_actuator_gears(model, max_gear_to_inertia_ratio=gear_clamp_ratio)

    # Reusing a compiled controller across seeds must not reuse its Python state.
    # Rebuild each side independently, even when both use the same controller.
    from mjarena.agents.policy_runtime import ActuatorPolicyAdapter

    if isinstance(red_policy_py, ActuatorPolicyAdapter):
        red_policy_py = red_policy_py.new_match()
    if isinstance(blue_policy_py, ActuatorPolicyAdapter):
        blue_policy_py = blue_policy_py.new_match()

    # Create the two contenders
    _trace(f"{trace_label or 'match'}: BotRuntime construction start", enabled=trace_enabled)
    red_rt  = BotRuntime(model=model, data=data, policy_callable=red_policy_py,  prefix="red_", history_len=action_history_len, obs_lookback=max_obs_lookback, action_lookback=max_action_lookback)
    blue_rt = BotRuntime(model=model, data=data, policy_callable=blue_policy_py, prefix="blue_", history_len=action_history_len, obs_lookback=max_obs_lookback, action_lookback=max_action_lookback)
    if inactivity_exempt_prefixes is None:
        # Unactuated baseline blocks cannot choose to move. An explicit list
        # still permits callers to select exemptions for diagnostic matches.
        inactivity_exempt_prefixes = [
            bot.prefix for bot in (red_rt, blue_rt) if bot.action_dim == 0
        ]
    _trace(f"{trace_label or 'match'}: BotRuntime construction done", enabled=trace_enabled)

    # Create the environment with the two contenders
    env_kwargs = dict(env_kwargs or {})

    # Resolve match_time -> max_steps BEFORE constructing the env, so the env
    # and the Match wrapper share one step cap. (Otherwise the env keeps its
    # own default max_steps and truncates long matches with "timeout".)
    # The control timestep is resolved the same way SumoEnv resolves it:
    # explicit control_timestep override wins, else the contact-fidelity preset.
    if match_time is not None:
        fidelity = str(env_kwargs.get("contact_fidelity", "high")).lower()
        preset = (
            env_cls.HIGH_FIDELITY_CONTACT if fidelity == "high"
            else env_cls.LOW_FIDELITY_CONTACT
        )
        control_timestep = env_kwargs.get("control_timestep")
        if control_timestep is None:
            control_timestep = preset["control_timestep"]
        max_steps = int(match_time / float(control_timestep))

    _trace(f"{trace_label or 'match'}: SumoEnv construction start", enabled=trace_enabled)
    env = env_cls(
        model=model,
        data=data,
        xml_path=composed_xml,
        red_contender=red_rt,
        blue_contender=blue_rt,
        max_steps=max_steps,
        render_mode=("viewer" if use_gui else "offscreen"),
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        inactivity_min_displacement=inactivity_min_displacement,
        inactivity_exempt_prefixes=inactivity_exempt_prefixes,
        size_limits=size_limits,
        **env_kwargs,
    )
    _trace(f"{trace_label or 'match'}: SumoEnv construction done", enabled=trace_enabled)

    match = Match(
        env, red_rt, blue_rt, max_steps=max_steps, headless=not use_gui,
        seed=seed, score_function=score_function, observation_config=observation_config,
        match_time=match_time, game_context=game_context, initial_state=initial_state,
    )
    _trace(f"{trace_label or 'match'}: Match wrapper constructed", enabled=trace_enabled)

    # Run the match
    default_video_path = out_dir / "match.mp4"
    final_video_path = video_path if video_path else str(default_video_path)
    try:
        result = match.run(save_video=save_video,
                           video_path=final_video_path,
                           video_width=video_width,
                           video_height=video_height,
                           camera_mode=camera_mode,
                           quiet=quiet,
                           overlay_info=overlay_info,
                           trace_label=trace_label,
                           progress_callback=progress_callback,
                           rendering_flags=rendering_flags,
                           )
    finally:
        env.detailed_observations.close()
    _trace(f"{trace_label or 'match'}: run_match returning", enabled=trace_enabled)
    return result
