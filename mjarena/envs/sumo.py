from __future__ import annotations

import math
import os
import random
import mujoco
import numpy as np
from lxml import etree
import gymnasium as gym
from pathlib import Path
from gymnasium import spaces
from typing import Dict, Optional, List, Tuple

from mjarena.envs.utils import _etree, _find_spawn_and_ring, _extract_robot_subtrees, _get_robot_root_body, _collect_name_map, _apply_name_map, _recolor_robot_geoms, _validate_robot_composition
from mjarena.agents.runtime import BotRuntime, _body_belongs_to_contender
from mjarena.envs.history import DEFAULT_HISTORY_LEN as MAX_PREVIOUS_ACTIONS
def _get_physics_diagnostics():
    from mjarena.design_shop.tools.physics_diagnostics import get_physics_diagnostics
    return get_physics_diagnostics

SETTLE_STEPS = 50  # 50 × 0.00025s = 12.5ms — gentle settle onto ring surface


def _geom_z_bounds(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[float, float]:
    pos = np.array(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.array(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    gtype = int(model.geom_type[geom_id])
    size = np.array(model.geom_size[geom_id], dtype=np.float64)

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
        return pos[2] + r, pos[2] - r

    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius = float(size[0])
        half = float(size[1]) if size.size >= 2 else 0.0
        axis = mat[:, 2]
        end1 = pos + axis * half
        end2 = pos - axis * half
        max_end = max(end1[2], end2[2])
        min_end = min(end1[2], end2[2])
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # Capsule: hemispherical caps extend full radius in every direction
            return max_end + radius, min_end - radius
        # Cylinder: flat ends, radius extends perpendicular to axis
        # Z contribution = radius * sin(angle_between_axis_and_Z)
        axis_z_sq = axis[2] ** 2
        perp_radius = radius * np.sqrt(max(0.0, 1.0 - axis_z_sq))
        return max_end + perp_radius, min_end - perp_radius

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        hx = float(size[0]) if size.size >= 1 else 0.0
        hy = float(size[1]) if size.size >= 2 else 0.0
        hz = float(size[2]) if size.size >= 3 else 0.0
        ext_z = abs(mat[2, 0]) * hx + abs(mat[2, 1]) * hy + abs(mat[2, 2]) * hz
        return pos[2] + ext_z, pos[2] - ext_z

    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent_z = float(np.linalg.norm(mat[2] * size))
        return pos[2] + extent_z, pos[2] - extent_z

    if gtype in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            vert_count = int(model.mesh_vertnum[mesh_id])
            vert_adr = int(model.mesh_vertadr[mesh_id])
            verts = np.array(model.mesh_vert[vert_adr:vert_adr + vert_count], dtype=np.float64).reshape(vert_count, 3)
            world = verts @ mat.T + pos
            return float(world[:, 2].max()), float(world[:, 2].min())

    # Fallback: treat like sphere using first size entry
    default_radius = float(size[0]) if size.size else 0.0
    return pos[2] + default_radius, pos[2] - default_radius

def compose_sumo_model(
    env_xml: str,
    robot_red_xml: str,
    robot_blue_xml: str,
    out_path: str,
    red_prefix: str = "red_",
    blue_prefix: str = "blue_",
    height_offset: float = 0.25,
    randomize_spawn_3d: bool = False,
    use_material_palette: bool = True,
    beacon_size_m: float = 0.12,
    spawn_seed: int | None = None,
) -> str:
    """Compose a single MJCF model with ring + 2 robots placed at spawn sites.

    Position fairness (which side each robot starts on) is handled at runtime
    by SumoEnv._set_start_positions(seed), not here.

    If randomize_spawn_3d is True, both robots are placed at random positions
    on opposite halves of the ring (for 3D matches).
    """
    env_tree = _etree(env_xml)
    red_spawn, blue_spawn, ring_radius = _find_spawn_and_ring(env_tree)
    red_spawn = np.array(red_spawn, dtype=np.float64, copy=True)
    blue_spawn = np.array(blue_spawn, dtype=np.float64, copy=True)

    if randomize_spawn_3d and ring_radius > 0:
        import random, math
        # Use a local RNG for thread safety. When spawn_seed is provided
        # (e.g., from matchup seed), results are deterministic regardless
        # of parallel execution order. Otherwise fall back to global state.
        rng = random.Random(spawn_seed) if spawn_seed is not None else random.Random(random.getrandbits(64))
        # Randomize BOTH bots: pick a random angle for red on the ring
        # (at ~0.5× ring_radius), then place blue opposite with random offset.
        red_angle = rng.uniform(0, 2 * math.pi)
        red_r = rng.uniform(0.4, 0.6) * ring_radius
        red_spawn[0] = red_r * math.cos(red_angle)
        red_spawn[1] = red_r * math.sin(red_angle)
        # Blue: opposite half with random offset
        offset = rng.uniform(-math.pi / 2, math.pi / 2)
        blue_angle = red_angle + math.pi + offset
        blue_r = rng.uniform(0.4, 0.6) * ring_radius
        blue_spawn[0] = blue_r * math.cos(blue_angle)
        blue_spawn[1] = blue_r * math.sin(blue_angle)
        # z is set later from ring height logic

    red_tree = _etree(robot_red_xml)
    blue_tree = _etree(robot_blue_xml)

    # Also normalize older processed robots loaded directly by native/WASM
    # matches. Preserve material friction and mass already assigned to them.
    from mjarena.design_shop.utils import normalize_robot_collision_settings
    for robot_tree in (red_tree, blue_tree):
        normalize_robot_collision_settings(robot_tree.getroot())

    env_root = env_tree.getroot()

    # Use absolute path so composed.xml works even when copied to other locations
    # (e.g., --save-match-logs copies it to a different directory)
    env_dir_abs = str(Path(env_xml).resolve().parent)
    compiler = env_root.find("compiler")
    if compiler is None:
        compiler = etree.SubElement(env_root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("texturedir", env_dir_abs)
    compiler.set("meshdir", env_dir_abs)

    # Normalize any absolute/embedded prefixes inside <texture file="..."> etc.
    # We strip leading 'assets/' or 'mjarena/assets/' so the file becomes
    # relative to texturedir set above.
    def _strip_asset_prefix(path_str: str) -> str:
        p = path_str.replace("\\", "/")
        for prefix in ("mjarena/assets/", "assets/"):
            if p.startswith(prefix):
                return p[len(prefix):]
        return p
    for tex in env_root.findall(".//texture"):
        f = tex.get("file")
        if f:
            tex.set("file", _strip_asset_prefix(f))
    for mesh in env_root.findall(".//mesh"):
        f = mesh.get("file")
        if f:
            mesh.set("file", _strip_asset_prefix(f))
    for hf in env_root.findall(".//hfield"):
        f = hf.get("file")
        if f:
            hf.set("file", _strip_asset_prefix(f))

    # Provisional root height; exact geometry clearance is computed after assembly.
    spawn_height_offset = float(height_offset) + 0.02

    # Pull top-level holders (create if missing)
    def get_or_make(tag: str) -> etree._Element:
        nodes = env_root.findall(tag)
        if nodes:
            return nodes[0]
        n = etree.SubElement(env_root, tag)
        return n

    env_world = env_root.find("worldbody")
    if env_world is None:
        env_world = etree.SubElement(env_root, "worldbody")

    env_asset = get_or_make("asset")
    env_default = get_or_make("default")
    env_actuator = get_or_make("actuator")
    env_sensor = get_or_make("sensor")
    env_contact = get_or_make("contact")
    env_equality = get_or_make("equality")
    env_tendon = get_or_make("tendon")

    # Default classes form a separate namespace. Remap before moving any XML
    # subtrees, including references on unnamed geoms and nested defaults.
    for tree, prefix in ((red_tree, red_prefix), (blue_tree, blue_prefix)):
        classes = {d.get("class"): prefix + d.get("class")
                   for d in tree.getroot().iter("default") if d.get("class")}
        for element in tree.getroot().iter():
            for attr in ("class", "childclass"):
                value = element.get(attr)
                if value in classes:
                    element.set(attr, classes[value])

    # Extract sections from robots
    red_sections = _extract_robot_subtrees(red_tree)
    blue_sections = _extract_robot_subtrees(blue_tree)

    red_root = _get_robot_root_body(red_sections["worldbody"])
    blue_root = _get_robot_root_body(blue_sections["worldbody"])

    # Validate single-tree structure BEFORE prefixing, so any error names the
    # model's own elements (axle_fl, not red_axle_fl) and explains the dropped-body
    # cause instead of MuJoCo's cryptic 'unknown transmission target'.
    _validate_robot_composition(red_sections, red_root)
    _validate_robot_composition(blue_sections, blue_root)

    # A root name anchors runtime ownership. MJCF permits unnamed bodies;
    # give an unnamed root a stable name before applying the robot prefix.
    for tree, root in ((red_tree, red_root), (blue_tree, blue_root)):
        if not root.get("name"):
            occupied = {element.get("name") for element in tree.getroot().iter()}
            name = "__unnamed_root"
            while name in occupied:
                name += "_"
            root.set("name", name)

    # Prefix names and references for each robot
    red_map = _collect_name_map(red_tree.getroot(), red_prefix)
    _apply_name_map(red_sections["worldbody"], red_map)
    for tag in ["actuator", "sensor", "asset", "contact", "equality", "tendon"]:
        if tag in red_sections:
            _apply_name_map(red_sections[tag], red_map)
    # Also remap references inside <default> blocks (e.g., material="self")
    for d in red_tree.getroot().findall("default"):
        _apply_name_map(d, red_map)

    blue_map = _collect_name_map(blue_tree.getroot(), blue_prefix)
    _apply_name_map(blue_sections["worldbody"], blue_map)
    for tag in ["actuator", "sensor", "asset", "contact", "equality", "tendon"]:
        if tag in blue_sections:
            _apply_name_map(blue_sections[tag], blue_map)
    for d in blue_tree.getroot().findall("default"):
        _apply_name_map(d, blue_map)

    # Position roots
    def set_pos(el: etree._Element, pos: np.ndarray):
        el.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

    # If requested, lift spawns above the ring top so robots fall
    ring_geom = env_root.xpath(".//geom[@name='sumo_ring']")
    if ring_geom:
        geom_el = ring_geom[0]
        size = np.fromstring(geom_el.get("size"), sep=" ")
        pos_vals = np.fromstring(geom_el.get("pos", "0 0 0"), sep=" ")
        geom_type = (geom_el.get("type") or "").lower()
        if geom_type == "box":
            halfheight = float(size[2]) if size.size >= 3 else 0.0
        else:
            # Default: cylinder/capsule etc use second component as half-height
            halfheight = float(size[1]) if size.size >= 2 else 0.0
        ring_top_z = (pos_vals[2] if pos_vals.size >= 3 else 0.0) + halfheight
        target_red_height = ring_top_z + spawn_height_offset
        target_blue_height = ring_top_z + spawn_height_offset
        red_spawn = np.array([red_spawn[0], red_spawn[1], target_red_height], dtype=np.float64)
        blue_spawn = np.array([blue_spawn[0], blue_spawn[1], target_blue_height], dtype=np.float64)

    # Enforce planar spawn at y=0 for 2D (depth locked by joint layout).
    # In 3D with randomized spawn, keep the y values.
    if not randomize_spawn_3d:
        if red_spawn.size >= 2:
            red_spawn[1] = 0.0
        if blue_spawn.size >= 2:
            blue_spawn[1] = 0.0

    set_pos(red_root, red_spawn)
    set_pos(blue_root, blue_spawn)

    # Team colors — skip when material palette active (preserve material appearance)
    if not use_material_palette:
        team_red = (0.85, 0.20, 0.20, 1.0)
        team_blue = (0.20, 0.35, 0.85, 1.0)
        _recolor_robot_geoms(red_root, team_red)
        _recolor_robot_geoms(blue_root, team_blue)

    # Give unnamed roots a unique name so compiled subtree IDs can identify
    # every descendant geom, including those on unnamed child bodies.
    for prefix, root in ((red_prefix, red_root), (blue_prefix, blue_root)):
        if not root.get("name"):
            names = {body.get("name") for body in root.iter("body")}
            name = f"{prefix}__spawn_root"
            while name in names:
                name += "_"
            root.set("name", name)

    # Insert the robot roots; finish merging dependencies before measuring height.
    env_world.append(red_root)
    env_world.append(blue_root)

    # COM-tracking team beacons (mocap bodies for visual team identification)
    if use_material_palette:
        for beacon_name, rgba in [
            ("red_beacon", "0.85 0.2 0.2 0.8"),
            ("blue_beacon", "0.2 0.35 0.85 0.8"),
        ]:
            beacon_body = etree.SubElement(env_world, "body")
            beacon_body.set("name", beacon_name)
            beacon_body.set("mocap", "true")
            beacon_body.set("pos", "0 0 2")
            beacon_geom = etree.SubElement(beacon_body, "geom")
            beacon_geom.set("name", f"{beacon_name}_geom")
            beacon_geom.set("type", "sphere")
            beacon_geom.set("size", f"{beacon_size_m}")
            beacon_geom.set("rgba", rgba)
            beacon_geom.set("contype", "0")
            beacon_geom.set("conaffinity", "0")

    # Merge other sections
    def maybe_merge(tag: str, src_root: etree._Element, dst_root: etree._Element):
        if tag == "asset":
            # Deduplicate by (tag,name). Many robot XMLs repeat common assets
            # like textures/materials named 'grid', 'ball', 'self', etc.
            existing: set = set()
            for c in dst_root:
                nm = c.get("name")
                if nm is not None:
                    existing.add((c.tag, nm))
            for child in list(src_root):
                nm = child.get("name")
                key = (child.tag, nm) if nm is not None else (child.tag, id(child))
                if nm is None or key not in existing:
                    dst_root.append(child)
                    if nm is not None:
                        existing.add(key)
        else:
            for child in list(src_root):
                dst_root.append(child)

    for tag, holder in [("asset", env_asset), ("actuator", env_actuator), ("sensor", env_sensor), ("contact", env_contact), ("equality", env_equality), ("tendon", env_tendon)]:
        if tag in red_sections:
            maybe_merge(tag, red_sections[tag], holder)
        if tag in blue_sections:
            maybe_merge(tag, blue_sections[tag], holder)

    # Merge <default> sections using scoped classes to prevent leaking.
    # Each robot's defaults are wrapped in <default class="{prefix}robot">
    # and the robot root body gets childclass="{prefix}robot" so the
    # defaults only apply to that robot's subtree.
    def merge_defaults_scoped(
        robot_tree: etree._Element,
        prefix: str,
        robot_root: etree._Element,
        dst_root: etree._Element,
    ):
        defs = robot_tree.getroot().findall("default")
        if not defs:
            return

        scope_class = f"{prefix}robot"
        scope_el = etree.SubElement(dst_root, "default")
        scope_el.set("class", scope_class)

        for d in defs:
            if d.get("class"):
                scope_el.append(d)
            else:
                for child in list(d):
                    scope_el.append(child)

        if robot_root.get("childclass") is None:
            robot_root.set("childclass", scope_class)
        for holder in (env_actuator, env_tendon):
            for element in holder:
                if element.get("name", "").startswith(prefix) and element.get("class") is None:
                    element.set("class", scope_class)

    merge_defaults_scoped(red_tree, red_prefix, red_root, env_default)
    merge_defaults_scoped(blue_tree, blue_prefix, blue_root, env_default)

    # Ensure materials referenced by robots exist after prefixing. If a geom
    # uses material="self" (common in Unimal), that material must be present
    # in env_asset; it is defined in robots' <asset>. We already merge assets,
    # but if any material refs were prefixed via _apply_name_map, ensure the
    # geoms in robots reference the prefixed names as well (handled above by
    # applying name map inside defaults). No extra action needed here.

    # Apply contact rules to arena geoms, robot geoms, and inherited defaults.
    for geom in env_root.iter("geom"):
        if geom.get("geom") is None:
            geom.set("condim", "6")
            if geom.get("friction") is not None:
                sliding = geom.get("friction").split()[0]
                geom.set("friction", f"{sliding} 0.005 0.0001")
    for pair in env_root.iter("pair"):
        pair.set("condim", "6")
        pair.attrib.pop("friction", None)

    # Measure only after assets, defaults, tendons, and actuators are merged.
    # Compiling earlier can fail on unresolved materials/classes and leave large
    # robots intersecting the platform at the provisional root height.
    if ring_geom:
        try:
            trial_model = mujoco.MjModel.from_xml_string(
                etree.tostring(env_tree, pretty_print=True, encoding="unicode")
            )
            trial_data = mujoco.MjData(trial_model)
            mujoco.mj_forward(trial_model, trial_data)
            ring_id = mujoco.mj_name2id(trial_model, mujoco.mjtObj.mjOBJ_GEOM, "sumo_ring")
            ring_top_z, _ = _geom_z_bounds(trial_model, trial_data, ring_id)
            for prefix, root, spawn in (
                (red_prefix, red_root, red_spawn),
                (blue_prefix, blue_root, blue_spawn),
            ):
                root_id = mujoco.mj_name2id(
                    trial_model, mujoco.mjtObj.mjOBJ_BODY, root.get("name")
                )
                bottoms = [
                    _geom_z_bounds(trial_model, trial_data, gid)[1]
                    for gid in range(trial_model.ngeom)
                    if int(trial_model.body_rootid[trial_model.geom_bodyid[gid]]) == root_id
                ]
                if not bottoms or not np.isfinite(bottoms).all():
                    raise ValueError(f"No finite geometry bounds for {prefix}robot")
                spawn[2] += ring_top_z + 0.003 - min(bottoms)
                set_pos(root, spawn)
        except Exception as exc:
            raise ValueError(f"Cannot compute safe robot spawn heights: {exc}") from exc

    # Write composed model
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env_tree.write(out_path, pretty_print=True, encoding="utf-8", xml_declaration=True)
    return out_path

class SumoEnv(gym.Env):
    """
    Single-process, two-robot Sumo environment on a MuJoCo ring.

    Observations are not returned via the Gymnasium API directly because
    this is a 2-agent setting. Instead, use the BattleBot helpers to query
    per-agent observations.
    """

    metadata = {"render_modes": ["viewer", "offscreen"], "render_fps": 60}

    # Preset options for contact/physics fidelity
    HIGH_FIDELITY_CONTACT = {
        "model_timestep": 0.00025,  # 4 kHz physics (40 substeps at 100 Hz control)
        "control_timestep": 0.01,   # 100 Hz control
    }
    LOW_FIDELITY_CONTACT = {
        "model_timestep": 0.005,   # 200 Hz physics (faster, coarser)
        "control_timestep": 0.02,  # 50 Hz control
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        xml_path: str,
        red_contender: BotRuntime,
        blue_contender: BotRuntime,
        control_timestep: Optional[float] = None,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
        contact_fidelity: str = "high",  # "low" or "high"
        apply_n_repeated_actions: int = 1,
        termination_mode: str = "end-when-hit-floor",  # "end-when-hit-floor" | "out-of-ring"
        reward_function: str = "default",
        viewer_camera_mode: Optional[str] = None,
        inactivity_timeout_seconds: Optional[float] = 10.0,
        inactivity_min_displacement: float = 0.5,
        inactivity_exempt_prefixes: Optional[List[str]] = None,
        size_limits: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__()
        self.model = model
        # Enforce at runtime too: preprocessed/composed XML must not bypass the
        # authoring validator, in either native MuJoCo or the WASM runner.
        from mjarena.design_shop.utils import validate_internal_actuation, validate_single_root_free_joint
        from mjarena.envs.utils import enforce_contact_settings
        actuation_errors = validate_internal_actuation(model)
        if actuation_errors:
            raise ValueError("Invalid robot actuation: " + "; ".join(actuation_errors))
        structure_errors = []
        for contender_prefix in (getattr(red_contender, "prefix", "red_"),
                                  getattr(blue_contender, "prefix", "blue_")):
            structure_errors += validate_single_root_free_joint(model, prefix=contender_prefix)
        if structure_errors:
            raise ValueError("Invalid robot structure: " + "; ".join(structure_errors))
        enforce_contact_settings(model)
        self.data = data
        self.xml_path = xml_path
        # Apply fidelity preset first, then allow explicit control_timestep override
        if str(contact_fidelity).lower() == "high":
            preset = self.HIGH_FIDELITY_CONTACT
        else:
            preset = self.LOW_FIDELITY_CONTACT

        # Set physics integration timestep on the MuJoCo model
        try:
            self.model.opt.timestep = float(preset["model_timestep"])
        except Exception:
            # Fallback in case the model is immutable; proceed with provided model setting
            pass

        # Control timestep used to compute substeps
        self.control_timestep = float(preset["control_timestep"]) if control_timestep is None else float(control_timestep)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._prev_observation = None
        self.apply_n_repeated_actions = apply_n_repeated_actions
        self.termination_mode = str(termination_mode).lower()
        self.reward_function = str(reward_function).lower()
        # Determine ring radius (read from geom size) and prefixes
        try:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sumo_ring")
        except Exception:
            raise Exception("Could not find sumo_ring geom")

        self._boundary_geom_id = int(gid)
        geom_type = int(self.model.geom_type[self._boundary_geom_id])
        self._boundary_shape = "cylinder"
        # Store half-extent along x as primary boundary limit; second entry kept for compatibility.
        self.ring_half_extents = np.zeros(2, dtype=np.float32)

        try:
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self._boundary_shape = "box"
                hx = float(self.model.geom_size[self._boundary_geom_id][0])
                depth = float(self.model.geom_size[self._boundary_geom_id][1])
                self.ring_half_extents = np.array([hx, depth], dtype=np.float32)
                self.ring_radius = hx
            else:
                # Treat cylinders and any other axially symmetric geom as circular ring
                self._boundary_shape = "cylinder"
                radius = float(self.model.geom_size[self._boundary_geom_id][0])
                self.ring_radius = radius
                self.ring_half_extents = np.array([radius, radius], dtype=np.float32)
        except Exception as exc:
            raise Exception("Could not infer boundary extents from geom 'sumo_ring'") from exc

        geom_pos = self.model.geom_pos[self._boundary_geom_id]
        if self._boundary_shape == "box":
            halfheight = float(self.model.geom_size[self._boundary_geom_id][2])
        else:
            halfheight = float(self.model.geom_size[self._boundary_geom_id][1])
        self.ring_top_z = float(geom_pos[2] + halfheight)

        # Cache outside floor geom id (if present)
        self.outside_floor_gid = -1
        try:
            self.outside_floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "outside_floor")
        except Exception:
            self.outside_floor_gid = -1

        nlight = getattr(self.model, "nlight", 0)
        if nlight > 0:
            self._original_light_diffuse = np.array(self.model.light_diffuse, dtype=np.float64, copy=True)
            self._original_light_specular = np.array(self.model.light_specular, dtype=np.float64, copy=True)
        else:
            self._original_light_diffuse = None
            self._original_light_specular = None
        self._winner_light_override = False

        self.red_contender = red_contender
        self.blue_contender = blue_contender
        self._root_body_ids: Dict[str, int] = {}
        self._floor_contact_geom_ids: Dict[str, set[int]] = {}
        self._contender_geom_ids: Dict[str, List[int]] = {}
        self._collect_root_body_ids()
        self._collect_contender_geom_ids()

        # COM-tracking beacons (mocap bodies, if present)
        self._beacon_height = 1.5
        self._red_mocap_id = -1
        self._blue_mocap_id = -1
        self._red_root_body_id = -1
        self._blue_root_body_id = -1
        try:
            red_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_beacon")
            blue_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_beacon")
            if red_bid >= 0 and blue_bid >= 0:
                # Find mocap IDs by scanning model.body_mocapid
                self._red_mocap_id = int(self.model.body_mocapid[red_bid])
                self._blue_mocap_id = int(self.model.body_mocapid[blue_bid])
                # Cache root body IDs for subtree_com
                red_prefix = getattr(self.red_contender, "prefix", "red_")
                blue_prefix = getattr(self.blue_contender, "prefix", "blue_")
                self._red_root_body_id = self._root_body_ids.get(red_prefix, -1)
                self._blue_root_body_id = self._root_body_ids.get(blue_prefix, -1)
        except Exception:
            pass

        # Inactivity tracking
        self._inactivity_min_displacement = inactivity_min_displacement
        self._inactivity_window_steps = (
            int(inactivity_timeout_seconds / self.control_timestep)
            if inactivity_timeout_seconds is not None else None
        )
        self._inactivity_exempt: set = set(inactivity_exempt_prefixes or [])

        from collections import deque
        # Include both endpoints: 1,001 samples span 10 s at 100 Hz.
        history_size = self._inactivity_window_steps + 1 if self._inactivity_window_steps is not None else 1
        self._red_com_history: deque = deque(maxlen=history_size)
        self._blue_com_history: deque = deque(maxlen=history_size)
        self._inactivity_timers = {"red": 0.0, "blue": 0.0}

        # Runtime size limit: root-frame extents (x, y, z) a robot may not exceed.
        # None disables the rule entirely (no measurement, no reporting).
        self._size_limits: Optional[Tuple[float, float, float]] = (
            tuple(float(v) for v in size_limits) if size_limits is not None else None
        )
        self._size_violation_loser: Optional[str] = None  # "red", "blue", "both"
        self._last_robot_extents: Optional[Dict[str, List[float]]] = None

        # QACC instability tracking
        self._qacc_warning_steps: list[int] = []
        self._qacc_terminated: bool = False
        self._qacc_loser: Optional[str] = None  # "red", "blue", or "both"
        self._qacc_dof_index: Optional[int] = None  # DOF MuJoCo blamed
        self._qacc_body_name: Optional[str] = None  # body that owns that DOF
        self._prev_qacc_warning_count: int = 0
        self._qacc_gear_diagnostics: list[dict] = []  # per-actuator gear/inertia info

        # Store spawn positions for swapping
        self._find_spawn_info()

        # Set up viewer if available; otherwise fallback to headless
        self.viewer = None
        if render_mode == "viewer":
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                raise RuntimeError(f"Could not start MuJoCo viewer, running headless: {e}")
            
        if self.viewer is not None:
            with self.viewer.lock():  # lock while editing camera
                try:
                    self.viewer.cam.orthographic = False
                except AttributeError:
                    pass
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                try:
                    self.viewer.cam.fixedcamid = -1
                except AttributeError:
                    pass
                mode = (viewer_camera_mode or "topdown").lower()
                if mode == "side":
                    try:
                        # Center view on arena and pull back for a broad side shot
                        self.viewer.cam.lookat[:] = self.model.stat.center
                    except Exception:
                        pass
                    self.viewer.cam.distance = float(self.model.stat.extent) * 1.8
                    self.viewer.cam.azimuth = 0
                    self.viewer.cam.elevation = -12
                else:
                    # Default: top-down
                    self.viewer.cam.distance = float(self.model.stat.extent) * 1.1
                    self.viewer.cam.azimuth = 45
                    self.viewer.cam.elevation = -90

    # --- boundary helpers ----------------------------------------------------
    def closest_distance_to_boundary(self, position: np.ndarray, use_3d: bool = False) -> float:
        """
        Signed distance from a position to the boundary.

        Positive = inside the ring, negative = outside the ring.

        For 2D mode: only considers X distance (Y is locked).
        For 3D mode: considers circular XY distance from center.

        Args:
            position: Position array [x, y, z] (uses x and optionally y)
            use_3d: If True, use circular XY distance check

        Returns:
            Signed distance to boundary (positive = inside, negative = outside)
        """
        pos = np.asarray(position, dtype=np.float32)
        x = float(pos[0]) if pos.size else 0.0

        # Check if any robot is 3D (freejoint)
        is_3d = use_3d or (hasattr(self, "_is_3d_robot") and any(self._is_3d_robot.values()))

        if self._boundary_shape == "box":
            hx = float(self.ring_half_extents[0])
            # dx = hx - |x|: positive inside, negative outside
            return float(hx - abs(x))

        if is_3d and pos.size >= 2:
            # 3D mode: circular distance from center
            y = float(pos[1])
            dist_from_center = np.sqrt(x * x + y * y)
            return float(self.ring_radius - dist_from_center)

        # 2D mode: only X distance (Y is locked at 0)
        return float(self.ring_radius - abs(x))

    def get_contact_info(self, prefix: str) -> tuple:
        """Get contact information for a robot.

        Args:
            prefix: Robot prefix (e.g., "red_" or "blue_")

        Returns:
            Tuple of (opponent_contact, opponent_force, ground_contact):
            - opponent_contact: bool, is this robot touching the opponent?
            - opponent_force: float, total contact force magnitude (N)
            - ground_contact: bool, is any of this robot's geoms touching the ring?
        """
        my_geoms = set(self._contender_geom_ids.get(prefix, []))
        # Find opponent prefix
        red_prefix = getattr(self.red_contender, "prefix", "red_")
        blue_prefix = getattr(self.blue_contender, "prefix", "blue_")
        opp_prefix = blue_prefix if prefix == red_prefix else red_prefix
        opp_geoms = set(self._contender_geom_ids.get(opp_prefix, []))

        opponent_contact = False
        opponent_force = 0.0
        ground_contact = False
        ring_gid = self._boundary_geom_id

        force_buf = np.zeros(6, dtype=np.float64)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            g1_mine = g1 in my_geoms
            g2_mine = g2 in my_geoms

            if not g1_mine and not g2_mine:
                continue

            other = g2 if g1_mine else g1

            # Robot-vs-robot contact
            if other in opp_geoms:
                opponent_contact = True
                mujoco.mj_contactForce(self.model, self.data, i, force_buf)
                # mj_contactForce returns force (N), then torque (N·m).
                opponent_force += float(np.linalg.norm(force_buf[:3]))

            # Robot-vs-ring-surface contact
            if other == ring_gid:
                ground_contact = True

        return opponent_contact, opponent_force, ground_contact

    def _collect_contender_geom_ids(self) -> None:
        self._contender_geom_ids = {}
        self._contender_total_mass: Dict[str, float] = {}
        if self.model is None:
            return
        prefixes = (
            getattr(self.red_contender, "prefix", None),
            getattr(self.blue_contender, "prefix", None),
        )
        for prefix in prefixes:
            if not prefix:
                continue
            geoms_for_contender: List[int] = []
            for gid in range(self.model.ngeom):
                body_id = int(self.model.geom_bodyid[gid])
                if self._body_belongs_to_contender(body_id, prefix):
                    geoms_for_contender.append(gid)
            if geoms_for_contender:
                self._contender_geom_ids[prefix] = geoms_for_contender
            # Cache total robot mass for this prefix (used by thres-50 score function)
            total = sum(
                float(self.model.body_mass[bid])
                for bid in range(self.model.nbody)
                if self._body_belongs_to_contender(bid, prefix)
            )
            self._contender_total_mass[prefix] = total

    def _body_belongs_to_contender(self, body_id: int, prefix: str) -> bool:
        """Return True for physical robot bodies, excluding team-marker mocaps."""
        return _body_belongs_to_contender(self.model, body_id, prefix)

    def is_out_of_bounds(self, position: np.ndarray, *, prefix: Optional[str] = None) -> bool:
        """Check if a robot is out of bounds.

        For 2D robots: only checks X distance.
        For 3D robots: checks circular XY distance from center.
        """
        # Check if any robot is 3D (freejoint)
        is_3d = hasattr(self, "_is_3d_robot") and any(self._is_3d_robot.values())

        if prefix:
            geom_ids = self._contender_geom_ids.get(prefix, [])
            if geom_ids:
                for gid in geom_ids:
                    geom_pos = self.data.geom_xpos[gid]
                    center_x = float(geom_pos[0])
                    center_y = float(geom_pos[1]) if is_3d else 0.0
                    radius = float(self.model.geom_rbound[gid]) if hasattr(self.model, "geom_rbound") else 0.0
                    if radius <= 0.0:
                        size = np.array(self.model.geom_size[gid], dtype=np.float64)
                        radius = float(size[0]) if size.size else 0.0
                    if self._boundary_shape == "box":
                        hx = float(self.ring_half_extents[0])
                        intersects = abs(center_x) <= (hx + radius)
                    elif is_3d:
                        # 3D: circular distance check
                        dist_from_center = np.sqrt(center_x * center_x + center_y * center_y)
                        intersects = dist_from_center <= (self.ring_radius + radius)
                    else:
                        # 2D: only X distance
                        intersects = abs(center_x) <= (self.ring_radius + radius)
                    if intersects:
                        return False
                return True

        pos = np.asarray(position, dtype=np.float32)
        x = float(pos[0]) if pos.size else 0.0

        if self._boundary_shape == "box":
            hx = float(self.ring_half_extents[0])
            return bool(abs(x) > hx)

        if is_3d and pos.size >= 2:
            # 3D: circular distance from center
            y = float(pos[1])
            dist_from_center = np.sqrt(x * x + y * y)
            return bool(dist_from_center > self.ring_radius)

        # 2D: only X distance
        return bool(abs(x) > self.ring_radius)

    def _collect_root_body_ids(self) -> None:
        """Cache top-level body ids for each contender prefix."""
        self._root_body_ids = {}
        if self.model is None:
            return
        parents = np.asarray(self.model.body_parentid, dtype=int)
        for prefix in (
            getattr(self.red_contender, "prefix", None),
            getattr(self.blue_contender, "prefix", None),
        ):
            if not prefix:
                continue
            for body_id in range(1, self.model.nbody):
                if not self._body_belongs_to_contender(body_id, prefix):
                    continue
                if int(parents[body_id]) == 0:
                    self._root_body_ids[prefix] = body_id
                    break
        self._refresh_floor_contact_geom_ids()

    def _refresh_floor_contact_geom_ids(self) -> None:
        self._floor_contact_geom_ids = {}
        if self.model is None:
            return
        for prefix, body_id in self._root_body_ids.items():
            geoms_for_body: list[int] = []
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) == body_id:
                    geoms_for_body.append(gid)
            if geoms_for_body:
                self._floor_contact_geom_ids[prefix] = set(geoms_for_body)

    def _counts_as_floor_hit(self, contender_prefix: Optional[str], geom_id: int) -> bool:
        if not contender_prefix:
            return True
        allowed = self._floor_contact_geom_ids.get(contender_prefix)
        if not allowed:
            return True
        return geom_id in allowed

    def _restore_lighting(self) -> None:
        if getattr(self, "_original_light_diffuse", None) is None:
            return
        nlight = getattr(self.model, "nlight", 0)
        if nlight == self._original_light_diffuse.shape[0]:
            self.model.light_diffuse[:, :] = self._original_light_diffuse
        if (
            getattr(self.model, "light_specular", None) is not None
            and self._original_light_specular is not None
            and nlight == self._original_light_specular.shape[0]
        ):
            self.model.light_specular[:, :] = self._original_light_specular
        self._winner_light_override = False

    def set_winner_lighting(self, winner: str) -> None:
        """Hook for subclasses to update lighting cues when a winner is declared."""
        if getattr(self, "_original_light_diffuse", None) is None:
            return
        key = (winner or "").lower()
        color_map = {
            "red": np.array([1.0, 0.25, 0.25], dtype=np.float64),
            "blue": np.array([0.25, 0.45, 1.0], dtype=np.float64),
            "red+blue": np.array([0.85, 0.35, 0.95], dtype=np.float64),
        }
        target = color_map.get(key)
        if target is None:
            self._restore_lighting()
            return
        if getattr(self.model, "nlight", 0) == 0:
            return
        for lid in range(self.model.nlight):
            self.model.light_diffuse[lid, :3] = target
            if getattr(self.model, "light_specular", None) is not None:
                self.model.light_specular[lid, :3] = target
        self._winner_light_override = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._restore_lighting()
        self._collect_root_body_ids()
        self._qacc_warning_steps = []
        self._qacc_terminated = False
        self._qacc_loser = None
        self._qacc_dof_index = None
        self._qacc_body_name = None
        self._prev_qacc_warning_count = 0
        self._qacc_gear_diagnostics = []
        self._size_violation_loser = None
        self._last_robot_extents = None
        self._red_com_history.clear()
        self._blue_com_history.clear()
        self._inactivity_timers = {"red": 0.0, "blue": 0.0}

        # Set starting positions based on seed for fairness
        self._set_start_positions(seed if seed is not None else 0)

        # Settle: let robots land on ring under gravity with zero controls
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
        # Wipe velocities for clean match start
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.t = 0
        if hasattr(self, "detailed_observations"):
            self.detailed_observations.begin_interval()
            self.detailed_observations.capture_contacts()
        if self._inactivity_window_steps is not None:
            # Seed the window at round start, after settling, before the first step.
            self._red_com_history.append(self.red_contender.com_position.copy())
            self._blue_com_history.append(self.blue_contender.com_position.copy())
        return None, {}

    def _find_spawn_info(self) -> None:
        """Find and store spawn positions and joint info for both robots.

        Supports both 2D robots (with slide_x joints) and 3D robots (with freejoint).
        """
        red_prefix = getattr(self.red_contender, "prefix", "red_")
        blue_prefix = getattr(self.blue_contender, "prefix", "blue_")

        # Find root bodies and their positions
        red_body_id = self._root_body_ids.get(red_prefix, -1)
        blue_body_id = self._root_body_ids.get(blue_prefix, -1)

        if red_body_id < 0 or blue_body_id < 0:
            self._spawn_positions = None
            self._spawn_joints = None
            self._is_3d_robot = {"red": False, "blue": False}
            return

        # Get body base positions from model (these are fixed offsets)
        # For 2D (slide joints): only x-coordinate is needed
        red_base_pos = self.model.body_pos[red_body_id][0]  # x-coordinate for 2D
        blue_base_pos = self.model.body_pos[blue_body_id][0]  # x-coordinate for 2D
        # For 3D (freejoint): store full [x, y, z] position
        red_base_pos_3d = np.array(self.model.body_pos[red_body_id], dtype=np.float64).copy()
        blue_base_pos_3d = np.array(self.model.body_pos[blue_body_id], dtype=np.float64).copy()

        # Track if robots are 3D (freejoint) or 2D (slide joints)
        self._is_3d_robot = {"red": False, "blue": False}

        # First, check for freejoint (3D robots)
        red_freejoint_id = -1
        blue_freejoint_id = -1
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                jnt_body = self.model.jnt_bodyid[jid]
                if jnt_body == red_body_id:
                    red_freejoint_id = jid
                    self._is_3d_robot["red"] = True
                elif jnt_body == blue_body_id:
                    blue_freejoint_id = jid
                    self._is_3d_robot["blue"] = True

        # If both have freejoints, use 3D spawn logic with full [x, y, z] positions
        if red_freejoint_id >= 0 and blue_freejoint_id >= 0:
            self._spawn_positions = [red_base_pos_3d, blue_base_pos_3d]
            self._spawn_joints = {
                "red": (red_freejoint_id, self.model.jnt_qposadr[red_freejoint_id], red_base_pos_3d, "free"),
                "blue": (blue_freejoint_id, self.model.jnt_qposadr[blue_freejoint_id], blue_base_pos_3d, "free"),
            }
            return

        # Fall back to 2D slide joint logic
        red_jnt_id = -1
        blue_jnt_id = -1
        for jid in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if jnt_name.startswith(red_prefix) and "slide_x" in jnt_name:
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                    red_jnt_id = jid
            elif jnt_name.startswith(blue_prefix) and "slide_x" in jnt_name:
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                    blue_jnt_id = jid

        if red_jnt_id < 0 or blue_jnt_id < 0:
            self._spawn_positions = None
            self._spawn_joints = None
            return

        # Store spawn info: two positions and which joints control them
        self._spawn_positions = [red_base_pos, blue_base_pos]  # e.g., [-1.5, 1.5]
        self._spawn_joints = {
            "red": (red_jnt_id, self.model.jnt_qposadr[red_jnt_id], red_base_pos, "slide"),
            "blue": (blue_jnt_id, self.model.jnt_qposadr[blue_jnt_id], blue_base_pos, "slide"),
        }

    def _set_start_positions(self, seed: int) -> None:
        """Set starting positions for red and blue robots based on seed.

        Uses seed to randomly assign which side each robot starts on.
        Only horizontal positions are swapped; each robot keeps the spawn
        height computed for its own geometry.
        Supports both 2D robots (slide joints) and 3D robots (freejoint).

        Args:
            seed: Random seed - determines which robot starts on which side
        """
        if not hasattr(self, "_spawn_joints") or self._spawn_joints is None:
            return

        red_info = self._spawn_joints["red"]
        blue_info = self._spawn_joints["blue"]

        # Unpack joint info (4 elements for new format, 3 for legacy)
        if len(red_info) == 4:
            red_jnt_id, red_qpos_adr, red_base, red_type = red_info
            blue_jnt_id, blue_qpos_adr, blue_base, blue_type = blue_info
        else:
            # Legacy format without joint type
            red_jnt_id, red_qpos_adr, red_base = red_info
            blue_jnt_id, blue_qpos_adr, blue_base = blue_info
            red_type = blue_type = "slide"

        no_opponent = bool(getattr(self, "_stage2x_no_opponent", False))

        # Use seed to determine positions: odd seeds flip sides.
        # No-opponent Stage 2X tests keep red on its normal spawn and the
        # dummy blue body offstage; otherwise odd seeds can put red offstage.
        if no_opponent:
            red_target = self._spawn_positions[0]
            blue_target = self._spawn_positions[1]
        elif seed % 2 == 0:
            # Even seed: red at position 0 (left), blue at position 1 (right)
            red_target = self._spawn_positions[0]
            blue_target = self._spawn_positions[1]
        else:
            # Odd seed: red at position 1 (right), blue at position 0 (left)
            red_target = self._spawn_positions[1]
            blue_target = self._spawn_positions[0]

        # Random yaw offsets so bots don't always start facing each other.
        # Uses seed for deterministic replay.
        rng = random.Random(seed)
        red_yaw_offset = rng.uniform(-math.pi / 3, math.pi / 3)
        blue_yaw_offset = rng.uniform(-math.pi / 3, math.pi / 3)

        if red_type == "free":
            # Freejoint: qpos is [x, y, z, qw, qx, qy, qz]
            self.data.qpos[red_qpos_adr + 0] = red_target[0]  # x
            self.data.qpos[red_qpos_adr + 1] = red_target[1]  # y
            self.data.qpos[red_qpos_adr + 2] = red_base[2]  # robot-specific z
            # Quaternion: face roughly toward opponent with random offset.
            # In no-opponent tests, avoid facing the offstage placeholder.
            if no_opponent:
                yaw = red_yaw_offset
            else:
                yaw = math.atan2(blue_target[1] - red_target[1],
                                 blue_target[0] - red_target[0]) + red_yaw_offset
            self.data.qpos[red_qpos_adr + 3] = math.cos(yaw / 2)  # qw
            self.data.qpos[red_qpos_adr + 4] = 0.0                 # qx
            self.data.qpos[red_qpos_adr + 5] = 0.0                 # qy
            self.data.qpos[red_qpos_adr + 6] = math.sin(yaw / 2)  # qz
        else:
            # Slide joint: just set x offset
            self.data.qpos[red_qpos_adr] = red_target - red_base

        if blue_type == "free":
            # Freejoint: qpos is [x, y, z, qw, qx, qy, qz]
            self.data.qpos[blue_qpos_adr + 0] = blue_target[0]  # x
            self.data.qpos[blue_qpos_adr + 1] = blue_target[1]  # y
            self.data.qpos[blue_qpos_adr + 2] = blue_base[2]  # robot-specific z
            # Quaternion: face roughly toward opponent with random offset.
            if no_opponent:
                yaw = 0.0
            else:
                yaw = math.atan2(red_target[1] - blue_target[1],
                                 red_target[0] - blue_target[0]) + blue_yaw_offset
            self.data.qpos[blue_qpos_adr + 3] = math.cos(yaw / 2)  # qw
            self.data.qpos[blue_qpos_adr + 4] = 0.0                 # qx
            self.data.qpos[blue_qpos_adr + 5] = 0.0                 # qy
            self.data.qpos[blue_qpos_adr + 6] = math.sin(yaw / 2)  # qz
        else:
            # Slide joint: just set x offset
            self.data.qpos[blue_qpos_adr] = blue_target - blue_base
    
    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

    def check_inactivity(self) -> Optional[str]:
        """Lose when the rolling window's furthest two 3D COM positions are
        less than the displacement threshold apart. None keeps detection disabled.

        Each history retains the longest trailing segment with diameter below
        the threshold, capped to one full window. Its duration is the observed
        inactivity timer — what ``my_inactivity_timer`` /
        ``opponent_inactivity_timer`` report. A full window in this state is an
        inactivity loss.

        If both bots trip the rule on the same step the robot whose centre of
        mass is more than 1 cm higher wins; otherwise the round is a tie
        ("both"). There is no exemption for pinning: a pinner that stops moving
        loses like any other bot.
        """
        if self._inactivity_window_steps is None:
            return None

        timed_out = {}
        threshold_squared = self._inactivity_min_displacement ** 2
        for side, contender, history in (
            ("red", self.red_contender, self._red_com_history),
            ("blue", self.blue_contender, self._blue_com_history),
        ):
            if f"{side}_" in self._inactivity_exempt:
                self._inactivity_timers[side] = 0.0
                timed_out[side] = False
                continue

            position = contender.com_position.copy()
            history.append(position)
            offsets = np.asarray(history) - position
            separated = np.flatnonzero(
                np.einsum("ij,ij->i", offsets, offsets) >= threshold_squared
            )
            if separated.size:
                # Earlier pairs already satisfy the invariant. Only pairs with
                # the new point can violate it. Drop through the latest such
                # point to keep the longest suffix whose EVERY pair is close.
                # This is equivalent to checking all pairs, in O(window) per step.
                for _ in range(int(separated[-1]) + 1):
                    history.popleft()

            elapsed_steps = max(0, len(history) - 1)
            self._inactivity_timers[side] = elapsed_steps * self.control_timestep
            timed_out[side] = elapsed_steps >= self._inactivity_window_steps

        if timed_out["red"] and timed_out["blue"]:
            red_z = float(self.red_contender.com_position[2])
            blue_z = float(self.blue_contender.com_position[2])
            if red_z - blue_z > 0.01:
                return "blue"
            if blue_z - red_z > 0.01:
                return "red"
            return "both"
        if timed_out["red"]:
            return "red"
        if timed_out["blue"]:
            return "blue"
        return None

    def robot_extents(self, prefix: str) -> np.ndarray:
        """Axis spans (m) of every geom of one robot, measured in its root body's frame.

        The root frame, not the world axes: a legal 2.44 m robot yawed 45° spans
        3.45 m along world X, so a world-axis box would punish turning. In the
        root frame a rigid robot's extents never change and only a mechanism
        that actually extends can grow them.
        """
        from mjarena.design_shop.utils import root_frame_extents
        return root_frame_extents(
            self.model, self.data,
            self._contender_geom_ids.get(prefix, []),
            self._root_body_ids.get(prefix, -1),
        )

    def check_size_limit(self) -> Optional[str]:
        """Side(s) whose root-frame extents exceed the size box on this step."""
        self._last_robot_extents = None
        if self._size_limits is None:
            return None
        limits = np.asarray(self._size_limits, dtype=float)
        red_prefix = getattr(self.red_contender, "prefix", "red_")
        blue_prefix = getattr(self.blue_contender, "prefix", "blue_")
        extents = {"red": self.robot_extents(red_prefix), "blue": self.robot_extents(blue_prefix)}
        self._last_robot_extents = {side: ext.tolist() for side, ext in extents.items()}
        over = {side: bool(np.any(ext > limits + 1e-6)) for side, ext in extents.items()}
        if over["red"] and over["blue"]:
            return "both"
        return "red" if over["red"] else "blue" if over["blue"] else None

    def step(self, actions: Dict[str, np.ndarray]):
        """Apply per-agent actions and return the observations.

        actions: {"red": np.ndarray, "blue": np.ndarray}
        """

        info = {}
        # steps the contenders
        self.red_contender.apply_action(actions["red"])
        self.blue_contender.apply_action(actions["blue"])


        # Integrate physics for one control step duration
        timestep = float(self.model.opt.timestep)
        n_substeps = 1 if timestep <= 0 else max(1, int(round(self.control_timestep / timestep)))
        details = getattr(self, "detailed_observations", None)
        if details is not None:
            details.begin_interval()
        for _ in range(self.apply_n_repeated_actions):
            for _ in range(n_substeps):
                mujoco.mj_step(self.model, self.data)
                if details is not None:
                    details.capture_contacts(integrate=True)
        self.t += 1

        # Check for QACC NaN (physics instability detection)
        # MuJoCo auto-recovers from NaN qacc before returning, so
        # np.isnan(data.qacc) never fires. Instead check the warning
        # counter that MuJoCo increments reliably.
        qacc_warn_count = int(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        if qacc_warn_count > self._prev_qacc_warning_count:
            self._qacc_warning_steps.append(self.t)
            self._qacc_terminated = True
            self._prev_qacc_warning_count = qacc_warn_count
            # Determine which bot owns the unstable DOF
            dof_idx = int(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].lastinfo)
            self._qacc_dof_index = dof_idx
            if 0 <= dof_idx < self.model.nv:
                body_id = self.model.dof_bodyid[dof_idx]
                body_name = self.model.body(body_id).name
                self._qacc_body_name = body_name or None
                if body_name.startswith("red_"):
                    self._qacc_loser = "red"
                elif body_name.startswith("blue_"):
                    self._qacc_loser = "blue"
                else:
                    self._qacc_loser = "both"
            else:
                self._qacc_loser = "both"
            # Compute gear-to-inertia diagnostics on first QACC only, for the
            # bot that owns the bad DOF — or for both when nothing owns it.
            if not self._qacc_gear_diagnostics:
                colors = (
                    [self._qacc_loser]
                    if self._qacc_loser in ("red", "blue")
                    else ["red", "blue"]
                )
                for color in colors:
                    diag = _get_physics_diagnostics()(self.model, prefix=f"{color}_")
                    self._qacc_gear_diagnostics.extend(
                        {
                            "color": color,
                            "actuator": a["name"],
                            "gear": round(a["gear"], 1),
                            "body_inertia_min": a["body_inertia_min"],
                            "ratio": round(a["gear_to_inertia_ratio"], 0),
                        }
                        for a in diag["actuators"]
                    )
            elapsed = self.t * self.frame_timestep
            print(
                f"\033[33m[QACC] Detected physics instability at control step "
                f"{self.t} (t={elapsed:.4f}s, dof={dof_idx}, "
                f"body={self._qacc_body_name or 'unknown'}). Episode will terminate.\033[0m"
            )

        # Update COM-tracking beacons (mocap bodies hover above robot COM)
        if self._red_mocap_id >= 0 and self._red_root_body_id >= 0:
            red_com = self.data.subtree_com[self._red_root_body_id]
            self.data.mocap_pos[self._red_mocap_id] = red_com + np.array([0, 0, self._beacon_height])
        if self._blue_mocap_id >= 0 and self._blue_root_body_id >= 0:
            blue_com = self.data.subtree_com[self._blue_root_body_id]
            self.data.mocap_pos[self._blue_mocap_id] = blue_com + np.array([0, 0, self._beacon_height])

        # now get the observations
        red_closest_distance_to_ring = self.closest_distance_to_boundary(self.red_contender.com_position)
        blue_closest_distance_to_ring = self.closest_distance_to_boundary(self.blue_contender.com_position)
        distance_to_opponent = np.linalg.norm(self.red_contender.com_position - self.blue_contender.com_position)
        no_opponent = bool(getattr(self, "_stage2x_no_opponent", False))

        current_observation = {
            "red_com": self.red_contender.com_position,
            "blue_com": self.blue_contender.com_position,
            "t": np.array([self.t], dtype=np.int32),
            "red_previous_actions": np.array(self.red_contender.action_history.as_array(), dtype=np.float32).reshape(MAX_PREVIOUS_ACTIONS, self.red_contender.action_dim),
            "blue_previous_actions": np.array(self.blue_contender.action_history.as_array(), dtype=np.float32).reshape(MAX_PREVIOUS_ACTIONS, self.blue_contender.action_dim),
            "red_distance_to_opponent": np.array([distance_to_opponent], dtype=np.float32),
            "blue_distance_to_opponent": np.array([distance_to_opponent], dtype=np.float32),
            "red_closest_distance_to_ring": np.array([red_closest_distance_to_ring], dtype=np.float32),
            "blue_closest_distance_to_ring": np.array([blue_closest_distance_to_ring], dtype=np.float32),
        }
        self._prev_observation = current_observation.copy()

        # Determine termination conditions
        # 1) Out-of-ring once the entire contender leaves the platform
        red_prefix = getattr(self.red_contender, "prefix", None)
        blue_prefix = getattr(self.blue_contender, "prefix", None)
        red_out = self.is_out_of_bounds(self.red_contender.com_position, prefix=red_prefix)
        blue_out = self.is_out_of_bounds(self.blue_contender.com_position, prefix=blue_prefix)

        # 2) Hit outside_floor via contact detection
        # Any geom belonging to a robot touching the lava floor ends the game
        red_hit_floor = False
        blue_hit_floor = False
        red_floor_bodies: set = set()
        blue_floor_bodies: set = set()
        if self.outside_floor_gid >= 0:
            red_root_bid = self._root_body_ids.get(red_prefix, -1) if red_prefix else -1
            blue_root_bid = self._root_body_ids.get(blue_prefix, -1) if blue_prefix else -1
            for i in range(int(self.data.ncon)):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 == self.outside_floor_gid or g2 == self.outside_floor_gid:
                    other_gid = g2 if g1 == self.outside_floor_gid else g1

                    # Skip false-positive floor contacts from geoms that
                    # penetrated through the ring (soft-contact artifact).
                    # A geom inside the ring boundary AND near the ring
                    # surface cannot have legitimately reached the floor.
                    geom_pos = self.data.geom_xpos[other_gid]
                    dist_from_center = np.sqrt(float(geom_pos[0])**2 + float(geom_pos[1])**2)
                    geom_z = float(geom_pos[2])
                    if dist_from_center < (self.ring_radius - 0.1) and geom_z > (self.ring_top_z - 0.05):
                        continue

                    # Attribute the contact via body ancestry (walk the parent
                    # chain to a contender's root body). Geom names are NOT
                    # used: unnamed geoms would silently escape a name-prefix
                    # check.
                    body_id = int(self.model.geom_bodyid[other_gid])
                    owner = None
                    bid = body_id
                    while bid > 0:
                        if bid == red_root_bid:
                            owner = "red"
                            break
                        if bid == blue_root_bid:
                            owner = "blue"
                            break
                        bid = int(self.model.body_parentid[bid])
                    if owner is None:
                        # Fallback (root id not cached): body-name prefix check
                        if red_prefix and self._body_belongs_to_contender(body_id, red_prefix):
                            owner = "red"
                        elif blue_prefix and self._body_belongs_to_contender(body_id, blue_prefix):
                            owner = "blue"
                    if owner == "red":
                        red_hit_floor = True
                        red_floor_bodies.add(body_id)
                    elif owner == "blue":
                        blue_hit_floor = True
                        blue_floor_bodies.add(body_id)

        # Compute floor mass fractions (for thres-50 score function)
        red_floor_mass = sum(float(self.model.body_mass[bid]) for bid in red_floor_bodies)
        blue_floor_mass = sum(float(self.model.body_mass[bid]) for bid in blue_floor_bodies)
        red_total = self._contender_total_mass.get(red_prefix, 1.0) if red_prefix else 1.0
        blue_total = self._contender_total_mass.get(blue_prefix, 1.0) if blue_prefix else 1.0

        if no_opponent:
            blue_out = False
            blue_hit_floor = False
            blue_floor_bodies.clear()
            blue_floor_mass = 0.0

        info = {
            "red_out": red_out,
            "blue_out": blue_out,
            "red_hit_floor": red_hit_floor,
            "blue_hit_floor": blue_hit_floor,
            "red_floor_mass_frac": red_floor_mass / red_total if red_total > 0 else 0.0,
            "blue_floor_mass_frac": blue_floor_mass / blue_total if blue_total > 0 else 0.0,
            "red_closest_distance_to_ring": red_closest_distance_to_ring,
            "blue_closest_distance_to_ring": blue_closest_distance_to_ring,
            "distance_to_opponent": distance_to_opponent,
            "physics_unstable": self.physics_unstable,
            "qacc_warning_steps": self._qacc_warning_steps,
            "qacc_gear_diagnostics": self._qacc_gear_diagnostics,
        }

        reward = 0.0
        if self.reward_function == "floor":
            if self.outside_floor_gid >= 0:
                # Only end the bout once a contender actually contacts the outside floor.
                terminated = red_hit_floor or blue_hit_floor
            else:
                # Fallback when no explicit outside floor geom is defined.
                terminated = red_out or blue_out
        else:
            if self.termination_mode == "end-when-hit-floor":
                terminated = red_hit_floor or blue_hit_floor
            else:
                terminated = red_out or blue_out
        # QACC NaN = immediate termination (simulation is broken)
        if self._qacc_terminated:
            terminated = True
        truncated = (self.t >= self.max_steps) if self.max_steps is not None else False

        # Termination reason tracking
        info["termination_reason"] = None
        info["inactivity_loser"] = None
        info["qacc_loser"] = None

        # QACC instability — attribute to the bot that owns the bad DOF
        if self._qacc_terminated and info["termination_reason"] is None:
            info["termination_reason"] = "qacc"
            info["qacc_loser"] = self._qacc_loser

        # Runtime size limit — a robot that unfolds past the size box loses.
        # Outranks inactivity and instability; floor contact still overrides it below.
        size_loser = self.check_size_limit()
        self._size_violation_loser = size_loser
        info["size_violation_loser"] = size_loser
        info["robot_extents"] = self._last_robot_extents
        if size_loser is not None:
            terminated = True
            info["termination_reason"] = "size_violation"

        # Inactivity tracking (displacement-based)
        inactivity_loser = self.check_inactivity()
        if inactivity_loser:
            terminated = True
            info["inactivity_loser"] = inactivity_loser
            if info["termination_reason"] != "size_violation":
                info["termination_reason"] = "inactivity"
        info["inactivity_timers"] = dict(self._inactivity_timers)

        # Ring-out detection (existing floor-hit logic) — overrides QACC/size/inactivity
        if info.get("red_hit_floor") or info.get("blue_hit_floor"):
            info["termination_reason"] = "ring_out"

        # Timeout (max steps reached)
        if truncated and info["termination_reason"] is None:
            info["termination_reason"] = "timeout"

        return current_observation, reward, bool(terminated), bool(truncated), info

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "red_com": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "blue_com": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "red_previous_actions": spaces.Box(low=-1.0, high=1.0, shape=(MAX_PREVIOUS_ACTIONS, self.red_contender.action_dim), dtype=np.float32),
            "blue_previous_actions": spaces.Box(low=-1.0, high=1.0, shape=(MAX_PREVIOUS_ACTIONS, self.blue_contender.action_dim), dtype=np.float32),
            "red_distance_to_opponent": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "blue_distance_to_opponent": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "red_closest_distance_to_ring": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "blue_closest_distance_to_ring": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    @property
    def frame_timestep(self) -> float:
        """Simulated seconds one control step advances — the replay frame interval.

        One step holds the chosen action for `apply_n_repeated_actions` control
        periods, so this is what separates two logged frames, not control_timestep.
        """
        return float(self.control_timestep) * int(self.apply_n_repeated_actions)

    @property
    def physics_unstable(self) -> bool:
        """True if any QACC NaN was detected during this episode."""
        return len(self._qacc_warning_steps) > 0

    def maybe_sync_viewer(self):
        if self.render_mode == "viewer" and self.viewer is not None:
            self.viewer.sync()

    def maybe_close_viewer(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

    def _enforce_environment_constraints(self) -> None:
        """Hook for subclasses to clamp MuJoCo state after each physics step."""
        return None


__all__ = [
    "compose_sumo_model",
    "SumoEnv",
]
