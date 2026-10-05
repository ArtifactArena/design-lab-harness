"""Export MuJoCo match trajectory to a format Blender can consume.

Reads composed.xml + match_data.json (with qpos), computes per-geom
world transforms for every frame, and saves to a .npz file.

Usage:
    python -m mjarena.rendering.export_trajectory composed.xml match_data.json -s 0 -o trajectory.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# MuJoCo geom type constants
_GEOM_TYPE_NAMES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_HFIELD: "hfield",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


def _find_assets_dir() -> Path:
    """Find the mjarena/assets directory by walking up from this file."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / "mjarena" / "assets"
        if candidate.is_dir():
            return candidate
        candidate = p / "assets"
        if candidate.is_dir():
            return candidate
        p = p.parent
    raise FileNotFoundError("Cannot find mjarena/assets directory")


def _load_composed_model(composed_xml: Path) -> mujoco.MjModel:
    """Load a composed.xml, fixing mesh/texture paths if they don't resolve."""
    from lxml import etree
    tree = etree.parse(str(composed_xml))
    root = tree.getroot()
    compiler = root.find("compiler")

    needs_fix = False
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            dirpath = compiler.get(attr, "")
            if dirpath:
                resolved = (composed_xml.parent / dirpath).resolve()
                if not resolved.is_dir():
                    needs_fix = True
                    break

    if needs_fix:
        assets_dir = _find_assets_dir()
        logger.info(f"Fixing asset paths → {assets_dir}")
        if compiler is None:
            compiler = etree.SubElement(root, "compiler")
        compiler.set("meshdir", str(assets_dir))
        compiler.set("texturedir", str(assets_dir))
        xml_str = etree.tostring(root, encoding="unicode")
        return mujoco.MjModel.from_xml_string(xml_str)
    else:
        return mujoco.MjModel.from_xml_path(str(composed_xml))


def _export_aligned_mesh(model: mujoco.MjModel, mesh_id: int, mesh_name: str, out_dir: Path) -> Path:
    """Export MuJoCo's auto-aligned mesh vertices as an OBJ file."""
    vert_start = model.mesh_vertadr[mesh_id]
    vert_count = model.mesh_vertnum[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    face_count = model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    obj_path = out_dir / f"{mesh_name}.obj"
    with open(obj_path, "w") as f:
        f.write(f"# MuJoCo auto-aligned mesh: {mesh_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    logger.info(f"Exported aligned mesh: {mesh_name} ({vert_count} verts, {face_count} faces)")
    return obj_path


def _bot_root_body_ids(model: mujoco.MjModel) -> tuple[int, int] | None:
    """Return (red_body_id, blue_body_id) from freejoints red_root / blue_root, or None."""
    red_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_root")
    blue_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "blue_root")
    if red_j < 0 or blue_j < 0:
        return None
    red_bid = int(model.jnt_bodyid[red_j])
    blue_bid = int(model.jnt_bodyid[blue_j])
    if red_bid == blue_bid:
        return None
    return red_bid, blue_bid


def _extract_geom_metadata(model: mujoco.MjModel) -> list[dict]:
    """Extract shape, size, color, and material info for every geom."""
    geoms = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        geom_type = int(model.geom_type[gid])
        type_name = _GEOM_TYPE_NAMES.get(geom_type, f"unknown_{geom_type}")
        size = model.geom_size[gid].tolist()
        rgba = model.geom_rgba[gid].tolist()
        group = int(model.geom_group[gid])

        # Material name and rgba (material rgba overrides geom rgba when assigned)
        mat_id = int(model.geom_matid[gid])
        mat_name = ""
        if mat_id >= 0:
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) or ""
            mat_rgba = model.mat_rgba[mat_id].tolist()
            # Use material color if geom has default gray
            if rgba == [0.5, 0.5, 0.5, 1.0]:
                rgba = mat_rgba

        # For mesh geoms, get the mesh name
        mesh_name = ""
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id >= 0:
                mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""

        geoms.append({
            "id": gid,
            "name": name,
            "type": type_name,
            "type_id": geom_type,
            "size": size,
            "rgba": rgba,
            "group": group,
            "material": mat_name,
            "mesh_name": mesh_name,
        })
    return geoms


def _resolve_mesh_paths(model: mujoco.MjModel, composed_xml: Path) -> dict[str, str]:
    """Resolve mesh file paths from the compiled model and XML compiler directives."""
    from lxml import etree
    tree = etree.parse(str(composed_xml))
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = ""
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")

    # Resolve meshdir relative to composed.xml location
    if meshdir:
        meshdir_abs = (composed_xml.parent / meshdir).resolve()
    else:
        meshdir_abs = composed_xml.parent.resolve()

    # Fallback: if meshdir doesn't exist, use the known assets directory
    if not meshdir_abs.is_dir():
        try:
            meshdir_abs = _find_assets_dir()
        except FileNotFoundError:
            pass

    mesh_paths = {}
    for mesh_el in root.findall(".//mesh"):
        name = mesh_el.get("name", "")
        file_attr = mesh_el.get("file", "")
        if name and file_attr:
            full_path = meshdir_abs / file_attr
            if full_path.exists():
                mesh_paths[name] = str(full_path)
            else:
                logger.warning(f"Mesh file not found: {full_path} (mesh={name})")
    return mesh_paths


def _ypr_to_quat(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    """Convert yaw/pitch/roll (radians, Z-Y-X intrinsic) to MuJoCo [w, x, y, z] quaternion."""
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return float(w), float(x), float(y), float(z)


def _freejoint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    """Look up qpos index of a named freejoint. Raises KeyError if missing."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint '{name}' not in model")
    return int(model.jnt_qposadr[jid])


def _reconstruct_qpos_from_actions(
    model: mujoco.MjModel, seed_data: dict
) -> list[list[float]]:
    """Rebuild per-frame qpos when match_data.json lacks raw qpos arrays.

    Root-body trajectory (position + orientation) is clamped to the recorded
    values exactly — so the bots track their true paths without drift. Wheel
    spin and other articulated joints are integrated via MuJoCo using the
    recorded actuator actions, so visuals include wheel rotation.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    red_addr = _freejoint_qpos_addr(model, "red_root")
    blue_addr = _freejoint_qpos_addr(model, "blue_root")

    def _set_root(addr: int, pos, ori: dict):
        data.qpos[addr:addr + 3] = pos
        data.qpos[addr + 3:addr + 7] = _ypr_to_quat(
            float(ori["yaw"]), float(ori["pitch"]), float(ori["roll"])
        )

    red_positions = seed_data["red_positions"]
    blue_positions = seed_data["blue_positions"]
    red_orientations = seed_data["red_orientations"]
    blue_orientations = seed_data["blue_orientations"]

    _set_root(red_addr, red_positions[0], red_orientations[0])
    _set_root(blue_addr, blue_positions[0], blue_orientations[0])
    mujoco.mj_forward(model, data)

    actuator_ctrl_idx = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_ctrl_idx[name] = i

    n_steps = int(seed_data["num_steps"])
    red_actions = seed_data["red_actions"]
    blue_actions = seed_data["blue_actions"]

    control_dt = float(seed_data.get("control_dt", 0.01))
    physics_dt = float(model.opt.timestep)
    n_substeps = max(1, int(round(control_dt / physics_dt)))

    n_frames = min(len(red_positions), len(blue_positions), n_steps + 1)
    qpos_frames: list[list[float]] = [data.qpos.copy().tolist()]
    for step in range(1, n_frames):
        data.ctrl[:] = 0.0
        act_step = min(step - 1, len(red_actions) - 1, len(blue_actions) - 1)
        for actions in (red_actions[act_step], blue_actions[act_step]):
            for act_name, val in actions.items():
                idx = actuator_ctrl_idx.get(act_name)
                if idx is not None:
                    data.ctrl[idx] = float(val)
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        _set_root(red_addr, red_positions[step], red_orientations[step])
        _set_root(blue_addr, blue_positions[step], blue_orientations[step])
        mujoco.mj_forward(model, data)
        qpos_frames.append(data.qpos.copy().tolist())

    return qpos_frames


def export_trajectory(
    composed_xml: Path,
    match_data_json: Path,
    seed: int = 0,
    output_npz: Optional[Path] = None,
    red_name: str = "",
    blue_name: str = "",
    season: str = "",
    tournament: str = "",
) -> Path:
    """Export per-geom transforms for every frame to a .npz file.

    Args:
        composed_xml: Path to the composed MuJoCo XML.
        match_data_json: Path to match_data.json with qpos arrays.
        seed: Which seed to export (default 0).
        output_npz: Output path (default: same dir as match_data, trajectory_seed_N.npz).

    Returns:
        Path to the written .npz file.
    """
    # Load match data
    with open(match_data_json) as f:
        match_data = json.load(f)

    seed_key = f"seed_{seed}"
    if seed_key not in match_data:
        raise KeyError(f"Seed {seed} not found in {match_data_json}. Available: {[k for k in match_data if k.startswith('seed_')]}")

    seed_data = match_data[seed_key]

    # Load MuJoCo model — fix mesh/texture paths if they don't resolve
    model = _load_composed_model(composed_xml)
    data = mujoco.MjData(model)
    n_geoms = model.ngeom

    qpos_frames = seed_data.get("qpos", [])
    if not qpos_frames:
        if "red_actions" in seed_data and "initial_red_pos" in seed_data:
            logger.info("No qpos in match_data — reconstructing by re-simulating recorded actions")
            qpos_frames = _reconstruct_qpos_from_actions(model, seed_data)
        else:
            raise ValueError(
                f"No qpos or reconstructable actions in {match_data_json} for seed {seed}."
            )

    n_frames = len(qpos_frames)
    control_dt = seed_data.get("control_dt", 0.01)
    winner = seed_data.get("winner", "tie")
    winner_step = seed_data.get("winner_step", n_frames)
    logger.info(f"Exporting {n_frames} frames (seed={seed}, winner={winner}, dt={control_dt})")

    # Extract geom metadata (static)
    geom_meta = _extract_geom_metadata(model)
    mesh_paths = _resolve_mesh_paths(model, composed_xml)

    # For mesh geoms, export MuJoCo's auto-aligned vertices as OBJ files
    # so Blender gets the same geometry MuJoCo uses internally
    aligned_mesh_dir = output_npz.parent / "aligned_meshes"
    aligned_mesh_dir.mkdir(parents=True, exist_ok=True)
    for gm in geom_meta:
        if gm["type"] == "mesh" and gm["mesh_name"]:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, gm["mesh_name"])
            if mesh_id >= 0:
                aligned_path = _export_aligned_mesh(model, mesh_id, gm["mesh_name"], aligned_mesh_dir)
                gm["mesh_path"] = str(aligned_path)
            elif gm["mesh_name"] in mesh_paths:
                gm["mesh_path"] = mesh_paths[gm["mesh_name"]]

    # Compute per-frame geom transforms
    geom_xpos = np.zeros((n_frames, n_geoms, 3), dtype=np.float32)
    geom_xmat = np.zeros((n_frames, n_geoms, 9), dtype=np.float32)

    bot_ids = _bot_root_body_ids(model)
    bot_xpos: np.ndarray | None = None
    red_bid = blue_bid = -1
    if bot_ids is not None:
        red_bid, blue_bid = bot_ids
        bot_xpos = np.zeros((n_frames, 2, 3), dtype=np.float32)

    for i, qpos in enumerate(qpos_frames):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        geom_xpos[i] = data.geom_xpos
        geom_xmat[i] = data.geom_xmat.reshape(n_geoms, 9)
        if bot_xpos is not None:
            bot_xpos[i, 0] = data.xpos[red_bid]
            bot_xpos[i, 1] = data.xpos[blue_bid]

    # Output path
    if output_npz is None:
        output_npz = match_data_json.parent / f"trajectory_seed_{seed}.npz"

    # Save everything
    save_kw: dict = dict(
        geom_xpos=geom_xpos,
        geom_xmat=geom_xmat,
        geom_meta=json.dumps(geom_meta),
        mesh_paths=json.dumps(mesh_paths),
        n_frames=n_frames,
        n_geoms=n_geoms,
        control_dt=control_dt,
        fps=1.0 / control_dt,
        duration=n_frames * control_dt,
        winner=winner,
        winner_step=winner_step,
        red_name=red_name or "Red",
        blue_name=blue_name or "Blue",
        season=season,
        tournament=tournament,
        seed=seed,
    )
    if bot_xpos is not None:
        save_kw["bot_xpos"] = bot_xpos
    np.savez_compressed(output_npz, **save_kw)

    size_mb = output_npz.stat().st_size / 1e6
    logger.info(f"Saved {output_npz} ({size_mb:.1f} MB)")
    return output_npz


def main():
    parser = argparse.ArgumentParser(description="Export MuJoCo match trajectory for Blender rendering")
    parser.add_argument("composed_xml", type=Path, help="Path to composed.xml")
    parser.add_argument("match_data_json", type=Path, help="Path to match_data.json")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Seed to export (default: 0)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npz path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    export_trajectory(args.composed_xml, args.match_data_json, args.seed, args.output)


if __name__ == "__main__":
    main()
