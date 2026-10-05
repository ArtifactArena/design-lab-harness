"""Blender cinematic renderer for MuJoCo match trajectories.

Reads a trajectory.npz (from export_trajectory.py), builds a scene in Blender
with neutral PBR materials, studio lighting, optional depth of field, and
post-processing, then renders to an image sequence.

Run via Blender's Python:
    blender --background --python render_blender.py -- trajectory.npz /tmp/output [width] [height] [samples] [engine] [frame_step] [config_json]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# PBR folders next to this script: arena_texture/ (concrete walls), ground_texture/ (floor)
TEXTURES_ROOT = Path(__file__).resolve().parent

# --- Blender imports (only available inside Blender's Python) ---
# Soft-fail so bpy-free helpers (e.g. _add_scoreboard_overlay) stay importable
# from the orchestrator when running outside Blender. Full-pipeline functions
# that touch bpy will raise naturally on call.
try:
    import bpy
    import mathutils
except ImportError:
    bpy = None  # type: ignore[assignment]
    mathutils = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
# Render Config — all enhancements toggled independently
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Robot visuals (neutral PBR; no team paint overrides)
    "subdivision": False,
    "bevel": False,
    "team_colors": False,
    "rubber_texture": True,

    # Arena
    "floor_grid": False,
    "ring_edge_glow": False,
    "gradient_background": True,

    # Camera & post
    "depth_of_field": True,
    "color_grading": True,
    "vignette": False,
    "bloom": True,
    "saturation": 2,

    # Dynamic effects
    "contact_sparks": False,
    "slow_motion": False,
    "bot_trails": False,

    # Small world-up team tint above each bot (body stays neutral PBR)
    "bot_team_markers": True,

    # Overlay
    "scoreboard": False,
}

PRESETS = {
    "minimal": {k: False for k in DEFAULT_CONFIG},
    "standard": dict(DEFAULT_CONFIG),
    "hero": {**{k: True for k in DEFAULT_CONFIG}, **{}},
}


# ═══════════════════════════════════════════════════════════════════
# Optional team tint (off by default — bots use neutral PBR below)
# ═══════════════════════════════════════════════════════════════════

# Slightly above (0,0,0) so Principled stays well-behaved; reads as black on display.
_RUBBER_ALBEDO: tuple[float, float, float, float] = (0.006, 0.006, 0.007, 1.0)

_RED_TEAM = {
    "chassis": (0.85, 0.2, 0.15, 1.0),
    "steel":   (0.75, 0.72, 0.72, 1.0),
    "rubber":  _RUBBER_ALBEDO,
    "default": (0.8, 0.25, 0.2, 1.0),
}

_BLUE_TEAM = {
    "chassis": (0.15, 0.25, 0.85, 1.0),
    "steel":   (0.72, 0.74, 0.8, 1.0),
    "rubber":  _RUBBER_ALBEDO,
    "default": (0.2, 0.35, 0.8, 1.0),
}


def _bot_neutral_base_color(name_l: str, mat_l: str, rgba: list[float]) -> tuple[float, float, float, float]:
    """Physical albedo presets (sRGB) — ignores MuJoCo paint RGB for hue."""
    if "foam" in mat_l:
        return (0.88, 0.86, 0.84, 1.0)
    if "plastic" in mat_l:
        return (0.76, 0.76, 0.78, 1.0)
    if "rubber" in mat_l:
        return _RUBBER_ALBEDO
    if "carbon" in mat_l or "fiber" in mat_l:
        return (0.09, 0.091, 0.093, 1.0)
    if "aluminum" in mat_l:
        return (0.84, 0.845, 0.85, 1.0)
    if "tungsten" in mat_l:
        return (0.62, 0.623, 0.628, 1.0)
    if "steel" in mat_l:
        # Brighter F0 tint so full-metallic steel does not read as charcoal under dim HDRI.
        return (0.91, 0.912, 0.918, 1.0)
    if "ballast" in name_l:
        return (0.86, 0.862, 0.865, 1.0)
    if "metal" in mat_l:
        return (0.78, 0.785, 0.79, 1.0)
    # Powder / composite body: neutral mid-gray from approximate MuJoCo luminance only
    L = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    g = 0.32 + min(max(L, 0.0), 0.55) * 0.45
    g = max(g, 0.48)
    return (g * 0.99, g, g * 1.01, 1.0)


def _team_indicator_rgba(is_red: bool) -> tuple[float, float, float, float]:
    """Saturated team tint for markers — a bit deeper than screen primaries."""
    if is_red:
        return (0.72, 0.04, 0.028, 1.0)
    return (0.028, 0.1, 0.78, 1.0)


def _red_blue_chassis_indices(geom_meta: list[dict]) -> tuple[int, int]:
    """Indices into geom_xpos for red/blue primary body geoms (same heuristic as camera)."""
    red_idx = blue_idx = None
    for i, gm in enumerate(geom_meta):
        nm = gm["name"].lower()
        if "red" in nm and "chassis" in nm:
            red_idx = i
        elif "blue" in nm and "chassis" in nm:
            blue_idx = i
    if red_idx is None or blue_idx is None:
        for i, gm in enumerate(geom_meta):
            if gm["group"] >= 3 or gm["rgba"][3] < 0.01:
                continue
            nm = gm["name"].lower()
            if red_idx is None and "red" in nm and gm["type"] in ("box", "cylinder"):
                red_idx = i
            elif blue_idx is None and "blue" in nm and gm["type"] in ("box", "cylinder"):
                blue_idx = i
    if red_idx is None:
        red_idx = 0
    if blue_idx is None:
        blue_idx = 0
    return red_idx, blue_idx


def _geom_z_top_half(gm: dict) -> float:
    """Approx +Z half-extent from geom center to upper surface (MuJoCo size convention)."""
    typ = gm.get("type", "box")
    sz = gm.get("size") or [0.4, 0.4, 0.25]
    try:
        if typ == "box":
            return float(sz[2])
        if typ == "cylinder":
            return float(sz[1])
        if typ == "capsule":
            return float(sz[1]) + float(sz[0])
        if typ == "sphere":
            return float(sz[0])
    except (TypeError, ValueError, IndexError):
        pass
    return 0.35


def _create_bot_team_marker(
    name: str,
    rgba: tuple[float, float, float, float],
    *,
    radius: float = 0.085,
    z_scale: float = 0.34,
):
    """Oblate sphere (squashed on world Z): reads like a flat bubble; emission-only = unshaded."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, segments=28, ring_count=14, location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (1.0, 1.0, z_scale)

    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (260, 0)
    em = nodes.new("ShaderNodeEmission")
    em.location = (0, 0)
    r, g, b = rgba[0], rgba[1], rgba[2]
    em.inputs["Color"].default_value = (r, g, b, 1.0)
    em.inputs["Strength"].default_value = 4.6
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _shade_auto_smooth_mesh(obj, 55.0)
    return obj


def _get_team_color(name: str, mat_name: str) -> tuple[float, ...] | None:
    """Get team color override based on geom/material name."""
    name_l = name.lower()
    mat_l = mat_name.lower()

    if name_l.startswith("red_"):
        palette = _RED_TEAM
    elif name_l.startswith("blue_"):
        palette = _BLUE_TEAM
    else:
        return None

    if "rubber" in mat_l:
        return palette["rubber"]
    if "steel" in mat_l or "aluminum" in mat_l or "metal" in mat_l or "tungsten" in mat_l:
        return palette["steel"]
    if "chassis" in mat_l or "chassis" in name_l:
        return palette["chassis"]
    if "ballast" in name_l:
        return palette["steel"]
    return palette["default"]


def _saturate_rgba(
    rgba: tuple[float, float, float, float],
    factor: float = 1.2,
) -> tuple[float, float, float, float]:
    """Push RGB away from luminance (simple saturation boost, clipped)."""
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]
    L = 0.299 * r + 0.587 * g + 0.114 * b
    nr = max(0.0, min(1.0, L + factor * (r - L)))
    ng = max(0.0, min(1.0, L + factor * (g - L)))
    nb = max(0.0, min(1.0, L + factor * (b - L)))
    return (nr, ng, nb, a)


# ═══════════════════════════════════════════════════════════════════
# Geom creation
# ═══════════════════════════════════════════════════════════════════

# Arena-interior geoms that Blender replaces with its own hand-built arena.
# Covers every variant in mjarena/assets/sumo_ring_env*.xml.
def _wheel_cylinder_geom(
    name: str, geom_type: str, size: list[float],
) -> tuple[str, list[float]]:
    """Use cylinders for drive wheels (MuJoCo is usually cylinder; fix sphere/capsule/mesh oddities)."""
    nl = name.lower()
    if "wheel" not in nl and "tire" not in nl:
        return geom_type, size
    sz = [float(s) for s in size]
    if geom_type == "sphere" and len(sz) >= 1:
        r = sz[0]
        half_h = max(0.035, min(r * 0.40, 0.22))
        return "cylinder", [r, half_h]
    if geom_type == "capsule" and len(sz) >= 2:
        return "cylinder", [sz[0], sz[1]]
    if geom_type == "mesh" and len(sz) >= 2:
        r = max(sz[0], sz[1], 0.01)
        half_h = sz[2] if len(sz) >= 3 else max(0.04, min(r * 0.38, 0.2))
        return "cylinder", [r, half_h]
    return geom_type, sz


_ARENA_GEOM_TAGS = (
    "sumo_ring",
    "outside_floor",
    "dohyo_body",
    "dohyo_top",
    "dohyo_bevel",
    "dohyo_boundary",
    "start_line",
    "ring_center_marker",
    "wall_left", "wall_right",
)


def _shade_auto_smooth_mesh(obj: bpy.types.Object, angle_deg: float = 50.0) -> None:
    """Angle-based auto smooth only — never uses bpy.ops.object.shade_smooth().

    Blender 4.2+: ``shade_auto_smooth(angle=…)``. Older: per-face smooth flags plus
    ``mesh.use_auto_smooth`` / ``auto_smooth_angle`` when those RNA paths exist.
    """
    if bpy is None or obj.type != "MESH":
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        ang_rad = math.radians(float(angle_deg))
    except (TypeError, ValueError):
        ang_rad = math.radians(50.0)

    op = getattr(bpy.ops.object, "shade_auto_smooth", None)
    if op is not None:
        for kwargs in ({"angle": ang_rad}, {"angle": float(angle_deg)}):
            try:
                op(**kwargs)
                return
            except TypeError:
                continue
            except Exception:
                break
        try:
            op()
            return
        except Exception:
            pass

    mesh = obj.data
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.update()
    try:
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = ang_rad
    except (AttributeError, TypeError):
        pass


def _arena_prism_bevel(obj: bpy.types.Object, width: float, *, angle_deg: float = 36.0) -> None:
    """Chamfer sharp creases on the octagonal arena prism (modifier, render-time)."""
    if bpy is None or obj.type != "MESH":
        return
    bev = obj.modifiers.new(name="ArenaBevel", type="BEVEL")
    bev.affect = "EDGES"
    bev.width = width
    bev.segments = 2
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(angle_deg)
    if hasattr(bev, "offset_type"):
        try:
            bev.offset_type = "OFFSET"
        except (TypeError, ValueError):
            pass


def _create_geom_object(meta: dict, cfg: dict) -> bpy.types.Object | None:
    """Create a Blender mesh object matching a MuJoCo geom."""
    geom_type, size = _wheel_cylinder_geom(meta["name"], meta["type"], meta["size"])
    name = meta["name"]
    group = meta["group"]
    rgba = meta["rgba"]

    if group >= 3 or rgba[3] < 0.01 or "beacon" in name.lower():
        return None

    # Skip arena geoms — we build clean versions in _setup_arena().
    # Exception: when KEEP_MJ_RING env var is set, keep the MuJoCo sumo_ring
    # (physics-faithful straight cylinder) — only its surrounding clutter
    # (start_lines, floor, markers, walls) is dropped.
    lname = name.lower()
    keep_mj_ring = bool(os.environ.get("KEEP_MJ_RING"))
    if keep_mj_ring and "sumo_ring" in lname:
        pass  # keep it
    elif any(tag in lname for tag in _ARENA_GEOM_TAGS):
        return None

    obj = None
    is_robot = name.lower().startswith("red_") or name.lower().startswith("blue_")

    if geom_type == "box":
        bpy.ops.mesh.primitive_cube_add(size=2)
        obj = bpy.context.active_object
        obj.scale = (size[0], size[1], size[2])

    elif geom_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0], segments=32, ring_count=16)
        obj = bpy.context.active_object

    elif geom_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=32)
        obj = bpy.context.active_object

    elif geom_type == "capsule":
        radius = size[0]
        half_len = size[1]
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=2 * half_len, vertices=32)
        cyl = bpy.context.active_object
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
        top = bpy.context.active_object
        top.location.z = half_len
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=16, ring_count=8)
        bot = bpy.context.active_object
        bot.location.z = -half_len
        bpy.ops.object.select_all(action='DESELECT')
        cyl.select_set(True)
        top.select_set(True)
        bot.select_set(True)
        bpy.context.view_layer.objects.active = cyl
        bpy.ops.object.join()
        obj = cyl

    elif geom_type == "plane":
        bpy.ops.mesh.primitive_plane_add(size=200)
        obj = bpy.context.active_object

    elif geom_type == "mesh":
        mesh_path = meta.get("mesh_path", "")
        if mesh_path and Path(mesh_path).exists():
            try:
                bpy.ops.wm.obj_import(filepath=mesh_path)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=mesh_path)
            obj = bpy.context.active_object
        else:
            return None
    else:
        return None

    if obj is None:
        return None

    obj.name = name

    # --- Modifiers for robot parts ---
    if is_robot and obj.type == 'MESH':
        if cfg.get("bevel") and geom_type == "box":
            bev = obj.modifiers.new(name="Bevel", type='BEVEL')
            bev.width = 0.008
            bev.segments = 2

        if cfg.get("subdivision"):
            sub = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            sub.levels = 0  # viewport
            sub.render_levels = 1  # render only

    return obj


def _link_roughness_noise(
    mat: bpy.types.Material,
    bsdf,
    rough_base: float,
    variation: float,
    noise_scale: float = 42.0,
    rough_floor: float = 0.02,
    rough_ceiling: float = 0.95,
) -> None:
    """Drive Principled Roughness with base + noise*variation (brushed metallic look)."""
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-920, -160)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-740, -160)
    mapping.inputs["Scale"].default_value = (2.8, 2.8, 2.8)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-520, -160)
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 9.0
    mult = nodes.new("ShaderNodeMath")
    mult.operation = "MULTIPLY"
    mult.location = (-300, -220)
    mult.inputs[1].default_value = variation
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.location = (-140, -180)
    add.inputs[1].default_value = rough_base
    lo = nodes.new("ShaderNodeMath")
    lo.operation = "MAXIMUM"
    lo.location = (20, -160)
    lo.inputs[1].default_value = rough_floor
    hi = nodes.new("ShaderNodeMath")
    hi.operation = "MINIMUM"
    hi.location = (180, -160)
    hi.inputs[1].default_value = rough_ceiling
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], mult.inputs[0])
    links.new(mult.outputs[0], add.inputs[0])
    links.new(add.outputs[0], lo.inputs[0])
    links.new(lo.outputs[0], hi.inputs[0])
    links.new(hi.outputs[0], bsdf.inputs["Roughness"])


def _principled_specular_boost(bsdf) -> None:
    sp = bsdf.inputs.get("Specular IOR Level")
    if sp is not None:
        sp.default_value = 1.0


def _material_dielectric(
    mat_name: str,
    rgb: tuple[float, float, float],
    roughness: float,
    sat_boost: float = 1.0,
):
    """Non-metallic arena / props — optional tiny saturation only (default neutral)."""
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        return mat
    r, g, b, _ = _saturate_rgba((*rgb, 1.0), sat_boost)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    return mat

def _material_arena_block_yellow(
    mat_name: str,
    *,
    color: tuple[float, float, float] | None = None,
    roughness: float | None = None,
    specular: float | None = None,
) -> bpy.types.Material:
    """Octagonal arena walls + cap: saturated yellow metallic shader (no textures).

    Supports optional overrides for callers that parameterize the ring look.
    """
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        return mat

    # Orange-leaning yellow (more red, less green than pure yellow).
    rgb = color if color is not None else (0.9, 0.55, 0.08)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.2
    bsdf.inputs["Roughness"].default_value = float(roughness) if roughness is not None else 0.22
    sp_value = float(specular) if specular is not None else None
    if sp_value is not None:
        sp_ior = bsdf.inputs.get("Specular IOR Level")
        if sp_ior is not None:
            sp_ior.default_value = sp_value
        sp_legacy = bsdf.inputs.get("Specular")
        if sp_legacy is not None:
            sp_legacy.default_value = sp_value

    em_col = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    em_str = bsdf.inputs.get("Emission Strength")
    if em_col is not None:
        em_col.default_value = (0.0, 0.02, 0.07, 1.0)
    if em_str is not None:
        em_str.default_value = 0.2

    ior_in = bsdf.inputs.get("IOR")
    if ior_in is not None:
        ior_in.default_value = 2.0

    # Principled v2 uses "Transmission Weight"; older builds use "Transmission".
    tw = bsdf.inputs.get("Transmission Weight")
    if tw is not None:
        tw.default_value = 0.3
    else:
        tr = bsdf.inputs.get("Transmission")
        if tr is not None:
            tr.default_value = 0.3

    print(f"Arena body: yellow block PBR ({mat_name})", flush=True)
    return mat


def _pbr_texture_paths(d: Path, *, base: str | None = None) -> dict[str, Path]:
    """Resolve a PBR set under a directory.

    If base is None, pick it from the first `*_color_*` file found.
    Expected outputs: color (sRGB), roughness/normal/ao (Non-Color).
    """
    if base is None:
        # Prefer jpg then png; sort for determinism.
        color_files = sorted(list(d.glob("*_color_*.jpg")) + list(d.glob("*_color_*.png")))
        if not color_files:
            return {}
        stem = color_files[0].stem
        # e.g. concrete_0002_color_2k -> concrete_0002
        base = stem.split("_color_")[0]

    normal_candidates = (
        d / f"{base}_normal_opengl_2k.png",
        d / f"{base}_normal_direct_2k.png",
        d / f"{base}_normal_directx_2k.png",
    )
    normal = next((p for p in normal_candidates if p.is_file()), normal_candidates[0])
    return {
        "color": next((p for p in (d / f"{base}_color_2k.jpg", d / f"{base}_color_2k.png") if p.is_file()), d / f"{base}_color_2k.jpg"),
        "roughness": next((p for p in (d / f"{base}_roughness_2k.jpg", d / f"{base}_roughness_2k.png") if p.is_file()), d / f"{base}_roughness_2k.jpg"),
        "normal": normal,
        "ao": next((p for p in (d / f"{base}_ao_2k.jpg", d / f"{base}_ao_2k.png") if p.is_file()), d / f"{base}_ao_2k.jpg"),
    }


def _arena_block_texture_paths() -> dict[str, Path]:
    d = TEXTURES_ROOT / "arena_texture"
    return _pbr_texture_paths(d, base="concrete_0002")


def _ground_texture_paths() -> dict[str, Path]:
    d = TEXTURES_ROOT / "ground_texture"
    # Auto-detect whatever texture set you dropped in ground_texture/.
    paths = _pbr_texture_paths(d, base=None)
    if paths:
        return paths
    # Fallback to the old default naming.
    return _pbr_texture_paths(d, base="plastic_0021")


def _mapping_vector_with_jitter(
    nodes,
    links,
    tc,
    mapping,
    *,
    base_socket: str,
    noise_scale: float,
    jitter_amp: tuple[float, float, float],
    loc_x: float = -1580.0,
    loc_y: float = -100.0,
) -> None:
    """Mapping input = base coords + low-frequency noise (weakens obvious square tiling)."""
    n_j = nodes.new("ShaderNodeTexNoise")
    n_j.location = (loc_x, loc_y)
    n_j.inputs["Scale"].default_value = noise_scale
    n_j.inputs["Detail"].default_value = 11.0
    n_j.inputs["Roughness"].default_value = 0.52
    links.new(tc.outputs["Object"], n_j.inputs["Vector"])

    amp = nodes.new("ShaderNodeRGB")
    amp.location = (loc_x + 180, loc_y + 200)
    ja, jb, jc = jitter_amp
    amp.outputs[0].default_value = (ja, jb, jc, 1.0)

    mul = nodes.new("ShaderNodeVectorMath")
    mul.location = (loc_x + 200, loc_y)
    mul.operation = "MULTIPLY"
    mul_a = mul.inputs.get("Vector") or mul.inputs[0]
    mul_b = mul.inputs.get("Vector_001") or mul.inputs[1]
    links.new(n_j.outputs["Color"], mul_a)
    links.new(amp.outputs["Color"], mul_b)

    add = nodes.new("ShaderNodeVectorMath")
    add.location = (loc_x + 380, loc_y + 120)
    add.operation = "ADD"
    add_a = add.inputs.get("Vector") or add.inputs[0]
    add_b = add.inputs.get("Vector_001") or add.inputs[1]
    links.new(tc.outputs[base_socket], add_a)
    mul_o = mul.outputs.get("Vector") or mul.outputs[0]
    links.new(mul_o, add_b)

    m_in = mapping.inputs.get("Vector") or mapping.inputs[0]
    add_o = add.outputs.get("Vector") or add.outputs[0]
    links.new(add_o, m_in)


def _material_arena_block_textured(
    mat_name: str,
    coord_scale: tuple[float, float, float] = (0.26, 0.26, 0.21),
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Material:
    """Octagon arena walls: concrete PBR from ``arena_texture/`` (BOX + Object coords).

    ``rotation`` is Euler XYZ in radians applied by the Mapping node (before jitter).
    """
    paths = _arena_block_texture_paths()
    need = ("color", "roughness", "normal", "ao")
    if not all(paths[k].is_file() for k in need):
        print(
            f"Arena body: missing maps under {TEXTURES_ROOT / 'arena_texture'} — dielectric fallback",
            flush=True,
        )
        return _material_dielectric(mat_name, (0.5, 0.495, 0.49), roughness=0.58, sat_boost=1.0)

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1180, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (760, 140)
    bsdf.inputs["Metallic"].default_value = 0.0

    tc = nodes.new("ShaderNodeTexCoord")
    tc.location = (-1460, 120)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-1220, 120)
    mapping.inputs["Scale"].default_value = coord_scale
    rot_in = mapping.inputs.get("Rotation")
    if rot_in is not None:
        rx, ry, rz = rotation
        rot_in.default_value = (rx, ry, rz)
    # Object + micro-jitter breaks perfect BOX period; softer blend hides projection seams.
    _mapping_vector_with_jitter(
        nodes,
        links,
        tc,
        mapping,
        base_socket="Object",
        noise_scale=2.1,
        jitter_amp=(0.011, 0.011, 0.011),
    )
    vec_out = mapping.outputs["Vector"]

    def _img(x: float, y: float, path: Path, colorspace: str):
        img = bpy.data.images.load(str(path.resolve()), check_existing=True)
        img.colorspace_settings.name = colorspace
        n = nodes.new("ShaderNodeTexImage")
        n.location = (x, y)
        n.image = img
        links.new(vec_out, n.inputs["Vector"])
        if hasattr(n, "projection"):
            try:
                n.projection = "BOX"
            except (TypeError, ValueError):
                pass
        if hasattr(n, "projection_blend"):
            try:
                n.projection_blend = 0.54
            except (TypeError, ValueError):
                pass
        return n

    col_tex = _img(-940, 400, paths["color"], "sRGB")
    rough_tex = _img(-940, 80, paths["roughness"], "Non-Color")
    nrm_tex = _img(-940, -240, paths["normal"], "Non-Color")
    ao_tex = _img(-940, -560, paths["ao"], "Non-Color")

    ao_lift = nodes.new("ShaderNodeMixRGB")
    ao_lift.location = (-440, 540)
    ao_lift.blend_type = "MIX"
    # Low Fac = less wash toward white so concrete grain reads on sunlit faces.
    ao_lift.inputs["Fac"].default_value = 0.08
    ao_white = nodes.new("ShaderNodeRGB")
    ao_white.location = (-620, 640)
    ao_white.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(ao_white.outputs["Color"], ao_lift.inputs["Color1"])
    links.new(ao_tex.outputs["Color"], ao_lift.inputs["Color2"])

    ao_boost = nodes.new("ShaderNodeMixRGB")
    ao_boost.location = (-260, 400)
    ao_boost.blend_type = "MULTIPLY"
    ao_boost.inputs["Fac"].default_value = 1.0
    links.new(col_tex.outputs["Color"], ao_boost.inputs["Color1"])
    links.new(ao_lift.outputs["Color"], ao_boost.inputs["Color2"])

    albedo_pop = nodes.new("ShaderNodeBrightContrast")
    albedo_pop.location = (-40, 400)
    albedo_pop.inputs["Bright"].default_value = -0.035
    albedo_pop.inputs["Contrast"].default_value = 0.14
    bc_in = albedo_pop.inputs.get("Color") or albedo_pop.inputs.get("Image")
    bc_out = albedo_pop.outputs.get("Color") or albedo_pop.outputs.get("Image")
    if bc_in is not None and bc_out is not None:
        links.new(ao_boost.outputs["Color"], bc_in)
        links.new(bc_out, bsdf.inputs["Base Color"])
    else:
        links.new(ao_boost.outputs["Color"], bsdf.inputs["Base Color"])

    sep_r = nodes.new("ShaderNodeSeparateColor")
    sep_r.location = (-560, 80)
    links.new(rough_tex.outputs["Color"], sep_r.inputs["Color"])
    r_chan = sep_r.outputs.get("Red") or sep_r.outputs[0]
    r_mul = nodes.new("ShaderNodeMath")
    r_mul.location = (-340, 100)
    r_mul.operation = "MULTIPLY"
    r_mul.inputs[1].default_value = 0.94
    links.new(r_chan, r_mul.inputs[0])
    r_add = nodes.new("ShaderNodeMath")
    r_add.location = (-160, 100)
    r_add.operation = "ADD"
    r_add.inputs[1].default_value = 0.03
    links.new(r_mul.outputs[0], r_add.inputs[0])
    r_clamp = nodes.new("ShaderNodeClamp")
    r_clamp.location = (40, 100)
    r_clamp.inputs["Min"].default_value = 0.05
    r_clamp.inputs["Max"].default_value = 1.0
    links.new(r_add.outputs[0], r_clamp.inputs["Value"])
    r_co = r_clamp.outputs.get("Result") or r_clamp.outputs[0]
    links.new(r_co, bsdf.inputs["Roughness"])

    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.location = (-560, -220)
    nmap.inputs["Strength"].default_value = 1.65
    links.new(nrm_tex.outputs["Color"], nmap.inputs["Color"])
    n_in = bsdf.inputs.get("Normal")
    if n_in is not None:
        links.new(nmap.outputs["Normal"], n_in)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    rz_deg = math.degrees(rotation[2]) if len(rotation) > 2 else 0.0
    print(
        f"Arena body: arena_texture concrete BOX+Object ({mat_name}) "
        f"scale={coord_scale} rot_z_deg={rz_deg:.1f}",
        flush=True,
    )
    return mat


def _material_arena_floor_textured(
    mat_name: str,
    coord_scale: tuple[float, float, float] = (44.0, 44.0, 1.0),
) -> bpy.types.Material:
    """Ground plane: plastic tile PBR from ``ground_texture/`` (UV + FLAT = aligned top/bottom)."""
    paths = _ground_texture_paths()
    need = ("color", "roughness", "normal", "ao")
    if not all(paths[k].is_file() for k in need):
        print(
            f"Arena floor: missing maps under {TEXTURES_ROOT / 'ground_texture'} — gray fallback",
            flush=True,
        )
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.012, 0.013, 0.018, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.96
            bsdf.inputs["Metallic"].default_value = 0.02
        return mat

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1100, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (720, 140)
    bsdf.inputs["Metallic"].default_value = 0.0
    # Dimmer specular response (plastic tile reads less mirror-like under key light).
    sp_ior = bsdf.inputs.get("Specular IOR Level")
    if sp_ior is not None:
        sp_ior.default_value = 0.2
    sp_legacy = bsdf.inputs.get("Specular")
    if sp_legacy is not None:
        sp_legacy.default_value = 0.28

    tc = nodes.new("ShaderNodeTexCoord")
    tc.location = (-1400, 120)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-1180, 120)
    mapping.inputs["Scale"].default_value = coord_scale
    links.new(tc.outputs["UV"], mapping.inputs["Vector"])
    vec_out = mapping.outputs["Vector"]

    def _img(x: float, y: float, path: Path, colorspace: str):
        img = bpy.data.images.load(str(path.resolve()), check_existing=True)
        img.colorspace_settings.name = colorspace
        n = nodes.new("ShaderNodeTexImage")
        n.location = (x, y)
        n.image = img
        links.new(vec_out, n.inputs["Vector"])
        if hasattr(n, "projection"):
            try:
                n.projection = "FLAT"
            except (TypeError, ValueError):
                pass
        return n

    col_tex = _img(-900, 340, paths["color"], "sRGB")
    rough_tex = _img(-900, 40, paths["roughness"], "Non-Color")
    nrm_tex = _img(-900, -260, paths["normal"], "Non-Color")
    ao_tex = _img(-900, -560, paths["ao"], "Non-Color")

    ao_lift = nodes.new("ShaderNodeMixRGB")
    ao_lift.location = (-420, 500)
    ao_lift.blend_type = "MIX"
    ao_lift.inputs["Fac"].default_value = 0.2
    ao_white = nodes.new("ShaderNodeRGB")
    ao_white.location = (-600, 620)
    ao_white.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(ao_white.outputs["Color"], ao_lift.inputs["Color1"])
    links.new(ao_tex.outputs["Color"], ao_lift.inputs["Color2"])

    ao_boost = nodes.new("ShaderNodeMixRGB")
    ao_boost.location = (-240, 340)
    ao_boost.blend_type = "MULTIPLY"
    ao_boost.inputs["Fac"].default_value = 1.0
    links.new(col_tex.outputs["Color"], ao_boost.inputs["Color1"])
    links.new(ao_lift.outputs["Color"], ao_boost.inputs["Color2"])

    darken_tint = nodes.new("ShaderNodeRGB")
    darken_tint.location = (-80, 500)
    # Stronger global darkening so the ground reads much darker.
    darken_tint.outputs[0].default_value = (0.14, 0.14, 0.16, 1.0)
    darken = nodes.new("ShaderNodeMixRGB")
    darken.location = (80, 340)
    darken.blend_type = "MULTIPLY"
    darken.inputs["Fac"].default_value = 1.0
    links.new(ao_boost.outputs["Color"], darken.inputs["Color1"])
    links.new(darken_tint.outputs["Color"], darken.inputs["Color2"])
    links.new(darken.outputs["Color"], bsdf.inputs["Base Color"])

    sep_r = nodes.new("ShaderNodeSeparateColor")
    sep_r.location = (-520, 40)
    links.new(rough_tex.outputs["Color"], sep_r.inputs["Color"])
    r_chan = sep_r.outputs.get("Red") or sep_r.outputs[0]
    r_mul = nodes.new("ShaderNodeMath")
    r_mul.location = (-320, 60)
    r_mul.operation = "MULTIPLY"
    r_mul.inputs[1].default_value = 1.06
    links.new(r_chan, r_mul.inputs[0])
    r_add = nodes.new("ShaderNodeMath")
    r_add.location = (-140, 60)
    r_add.operation = "ADD"
    r_add.inputs[1].default_value = 0.11
    links.new(r_mul.outputs[0], r_add.inputs[0])
    r_clamp = nodes.new("ShaderNodeClamp")
    r_clamp.location = (40, 60)
    r_clamp.inputs["Min"].default_value = 0.14
    r_clamp.inputs["Max"].default_value = 1.0
    links.new(r_add.outputs[0], r_clamp.inputs["Value"])
    r_co = r_clamp.outputs.get("Result") or r_clamp.outputs[0]
    links.new(r_co, bsdf.inputs["Roughness"])

    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.location = (-520, -240)
    nmap.inputs["Strength"].default_value = 1.15
    links.new(nrm_tex.outputs["Color"], nmap.inputs["Color"])
    n_in = bsdf.inputs.get("Normal")
    if n_in is not None:
        links.new(nmap.outputs["Normal"], n_in)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    print(f"Arena floor: ground_texture UV+FLAT ({mat_name}) scale={coord_scale}", flush=True)
    return mat


def _create_material(name: str, rgba: list[float], mat_name: str, cfg: dict) -> bpy.types.Material:
    """Create a Principled BSDF material (neutral PBR for robots; optional team tint if enabled)."""
    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    if not bsdf:
        return mat

    name_l = name.lower()
    mat_l = mat_name.lower()
    is_bot = name_l.startswith("red_") or name_l.startswith("blue_")

    if is_bot:
        color = _bot_neutral_base_color(name_l, mat_l, rgba)
    else:
        color = (rgba[0], rgba[1], rgba[2], 1.0)
        if cfg.get("team_colors"):
            team_color = _get_team_color(name, mat_name)
            if team_color:
                color = team_color
    is_floor = "floor" in name_l or "outside" in name_l
    is_rubber = "rubber" in mat_l
    is_steel = "steel" in mat_l or "metal" in mat_l or "chassis" in name_l or "tungsten" in mat_l
    is_platform = "platform" in name_l or "dohyo_body" in name_l
    is_top = "dohyo_top" in name_l
    # KEEP_MJ_RING: yellow plastic ring (matches Blender-built arena)
    is_mj_ring = "sumo_ring" in name_l and os.environ.get("KEEP_MJ_RING")

    bsdf.inputs["Base Color"].default_value = color

    if is_mj_ring:
        bsdf.inputs["Base Color"].default_value = (0.48, 0.48, 0.5, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.44
    elif is_floor:
        bsdf.inputs["Base Color"].default_value = (0.028, 0.03, 0.036, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.93
        bsdf.inputs["Metallic"].default_value = 0.02
    elif is_rubber:
        bsdf.inputs["Base Color"].default_value = _RUBBER_ALBEDO
        bsdf.inputs["Roughness"].default_value = 0.9
        bsdf.inputs["Metallic"].default_value = 0.0
        sp_ior = bsdf.inputs.get("Specular IOR Level")
        if sp_ior is not None:
            sp_ior.default_value = 0.2
        sp_legacy = bsdf.inputs.get("Specular")
        if sp_legacy is not None:
            sp_legacy.default_value = 0.52
    elif is_bot and ("foam" in mat_l or "plastic" in mat_l):
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.68
    elif is_bot and ("carbon" in mat_l or "fiber" in mat_l):
        bsdf.inputs["Metallic"].default_value = 0.78
        _link_roughness_noise(
            mat, bsdf, 0.26, 0.05, noise_scale=48.0, rough_floor=0.18, rough_ceiling=0.52
        )
        _principled_specular_boost(bsdf)
    elif is_bot and "aluminum" in mat_l:
        bsdf.inputs["Metallic"].default_value = 1.0
        _link_roughness_noise(
            mat, bsdf, 0.10, 0.035, noise_scale=90.0, rough_floor=0.03, rough_ceiling=0.26
        )
        _principled_specular_boost(bsdf)
    elif is_bot and "tungsten" in mat_l:
        bsdf.inputs["Metallic"].default_value = 1.0
        _link_roughness_noise(
            mat, bsdf, 0.085, 0.03, noise_scale=100.0, rough_floor=0.022, rough_ceiling=0.26
        )
        _principled_specular_boost(bsdf)
    elif is_bot and ("steel" in mat_l or "ballast" in name_l):
        bsdf.inputs["Metallic"].default_value = 1.0
        _link_roughness_noise(
            mat, bsdf, 0.022, 0.014, noise_scale=130.0, rough_floor=0.004, rough_ceiling=0.10
        )
        _principled_specular_boost(bsdf)
    elif is_bot and "metal" in mat_l:
        bsdf.inputs["Metallic"].default_value = 1.0
        _link_roughness_noise(
            mat, bsdf, 0.09, 0.04, noise_scale=100.0, rough_floor=0.025, rough_ceiling=0.28
        )
        _principled_specular_boost(bsdf)
    elif is_bot:
        bsdf.inputs["Metallic"].default_value = 0.88
        _link_roughness_noise(
            mat, bsdf, 0.18, 0.06, noise_scale=75.0, rough_floor=0.06, rough_ceiling=0.42
        )
        _principled_specular_boost(bsdf)
    elif is_steel:
        bsdf.inputs["Roughness"].default_value = 0.30
        bsdf.inputs["Metallic"].default_value = 0.7
    elif is_platform:
        bsdf.inputs["Roughness"].default_value = 0.50
        bsdf.inputs["Metallic"].default_value = 0.3
    elif is_top:
        bsdf.inputs["Roughness"].default_value = 0.45
        bsdf.inputs["Metallic"].default_value = 0.2
    else:
        bsdf.inputs["Roughness"].default_value = 0.60
        bsdf.inputs["Metallic"].default_value = 0.0

    if rgba[3] < 1.0:
        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'BLEND'
        bsdf.inputs["Alpha"].default_value = rgba[3]

    # Slight saturation on non-bot props only (bots stay neutral gray PBR).
    if not is_floor and not is_bot and not is_rubber:
        c = bsdf.inputs["Base Color"].default_value
        r, g, b, a = _saturate_rgba((c[0], c[1], c[2], c[3]), 1.04)
        bsdf.inputs["Base Color"].default_value = (r, g, b, a)

    # No fill-emission: metals read from arena + world + lights.
    if is_bot and not is_rubber and "foam" not in mat_l and "plastic" not in mat_l:
        es = bsdf.inputs.get("Emission Strength")
        if es is not None:
            es.default_value = 0.0

    return mat


def _xmat_to_euler(xmat_flat: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo 3x3 rotation matrix (row-major flat) to Blender Euler XYZ."""
    mat = mathutils.Matrix((
        (xmat_flat[0], xmat_flat[1], xmat_flat[2]),
        (xmat_flat[3], xmat_flat[4], xmat_flat[5]),
        (xmat_flat[6], xmat_flat[7], xmat_flat[8]),
    ))
    return mat.to_euler('XYZ')


# ═══════════════════════════════════════════════════════════════════
# World & Lighting
# ═══════════════════════════════════════════════════════════════════

def _setup_world(scene: bpy.types.Scene, cfg: dict):
    """Set up world environment."""
    world = bpy.data.worlds.new("CinematicWorld")
    scene.world = world
    if hasattr(world, 'use_nodes'):
        world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (600, 0)

    # World surface: RGB 28,39,44 (byte) → linear 0–1, strength 15 on Background.
    wr, wg, wb = 46.0 / 255.0, 53.0 / 255.0, 56.0 / 255.0
    world_bg_strength = 0.7

    if cfg.get("gradient_background"):
        # Neutral studio gradient.
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = world_bg_strength
        bg.location = (300, 0)

        gradient = nodes.new("ShaderNodeTexGradient")
        gradient.gradient_type = 'LINEAR'
        gradient.location = (-100, 0)

        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Location"].default_value = (0, 0, -0.3)
        # Lower scale ⇒ slower variation along Generated coords ⇒ less “striped” / repeated look.
        mapping.inputs["Scale"].default_value = (0.42, 0.42, 0.5)
        mapping.location = (-300, 0)

        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.location = (-500, 0)

        ramp = nodes.new("ShaderNodeValToRGB")
        # Ramp around requested world RGB (slightly lighter at top of ramp).
        ramp.color_ramp.elements[0].color = (wr * 0.85, wg * 0.85, wb * 0.85, 1.0)
        ramp.color_ramp.elements[1].color = (min(wr * 1.12, 1.0), min(wg * 1.12, 1.0), min(wb * 1.12, 1.0), 1.0)
        ramp.location = (100, 0)

        links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
        links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bg.inputs["Color"])
        links.new(bg.outputs["Background"], output.inputs["Surface"])
    else:
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = world_bg_strength
        bg.inputs["Color"].default_value = (wr, wg, wb, 1.0)
        bg.location = (300, 0)
        links.new(bg.outputs["Background"], output.inputs["Surface"])

def _setup_lighting(
    scene: bpy.types.Scene,
    *,
    ring_radius: float = 7.5,
    overhead_height: float = 20.0,
    arena_coverage_margin: float = 1.12,
):
    """Sun key + one overhead square area light sized to cover the arena floor (matches ``_setup_arena`` scale)."""
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 60.0))
    sun = bpy.context.active_object
    sun.name = "Sun_Fill"
    sun.rotation_euler = (math.radians(20), math.radians(-20), math.radians(12))
    sun.data.use_shadow = True
    if hasattr(sun.data, "energy"):
        sun.data.energy = 1.8
    if hasattr(sun.data, "angle"):
        sun.data.angle = math.radians(0.6)

    R = ring_radius * arena_coverage_margin
    side = 2.0 * R
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, overhead_height))
    area_ob = bpy.context.active_object
    area_ob.name = "Arena_Overhead_Area"
    area_ob.rotation_euler = (0.0, 0.0, 0.0)
    adata = area_ob.data
    adata.shape = "SQUARE"
    adata.size = side
    adata.use_shadow = True
    if hasattr(adata, "energy"):
        adata.energy = 10000.0


# ═══════════════════════════════════════════════════════════════════
# Ring edge glow
# ═══════════════════════════════════════════════════════════════════

def _setup_arena(
    ring_top_z: float = 2.0,
    ring_radius: float = 7.5,
    arena_color: tuple[float, float, float] | None = None,
    arena_roughness: float | None = None,
    arena_specular: float | None = None,
):
    """Build a vertical octagonal prism arena (constant radius — no flared base).

    `arena_color` / `arena_roughness` / `arena_specular` override the default
    bright-yellow ring material (used by the keyframe renderer for a muted ring
    that doesn't blow out under directional lighting).
    """
    import bmesh

    r_top = ring_radius
    r_bottom = r_top  # prism: vertical walls, bottom does not extend outward
    half_h = 1.0
    n_sides = 8
    angle_offset = math.pi / n_sides  # 22.5° so flat edge faces camera

    mesh = bpy.data.meshes.new("ArenaBody_mesh")
    bm = bmesh.new()

    # Top ring vertices
    top_verts = []
    for i in range(n_sides):
        angle = angle_offset + 2 * math.pi * i / n_sides
        v = bm.verts.new((r_top * math.cos(angle), r_top * math.sin(angle), half_h))
        top_verts.append(v)

    # Bottom ring (same radius — straight prism sides)
    bot_verts = []
    for i in range(n_sides):
        angle = angle_offset + 2 * math.pi * i / n_sides
        v = bm.verts.new((r_bottom * math.cos(angle), r_bottom * math.sin(angle), -half_h))
        bot_verts.append(v)

    # No top face — separate plastic disk on top (avoids z-fighting)

    # Bottom face (reversed winding)
    bm.faces.new(list(reversed(bot_verts)))

    # Side faces
    for i in range(n_sides):
        j = (i + 1) % n_sides
        bm.faces.new([top_verts[i], bot_verts[i], bot_verts[j], top_verts[j]])

    bm.to_mesh(mesh)
    bm.free()

    body = bpy.data.objects.new("Arena_Body", mesh)
    bpy.context.collection.objects.link(body)
    body.location = (0, 0, ring_top_z - half_h)  # bottom at z=0, top at ring_top_z

    body_mat = _material_arena_block_yellow(
        "mat_arena_body",
        color=arena_color,
        roughness=arena_roughness,
        specular=arena_specular,
    )
    body.data.materials.append(body_mat)
    _shade_auto_smooth_mesh(body, 42.0)
    _arena_prism_bevel(body, 0.042, angle_deg=36.0)

    # Top cap: exact same octagon radius as prism (flush vertical silhouette)
    r_top_visual = r_top
    top_mesh = bpy.data.meshes.new("ArenaTop_mesh")
    bm2 = bmesh.new()
    top_verts2 = []
    for i in range(n_sides):
        angle = angle_offset + 2 * math.pi * i / n_sides
        v = bm2.verts.new((r_top_visual * math.cos(angle), r_top_visual * math.sin(angle), 0))
        top_verts2.append(v)
    bm2.faces.new(top_verts2)
    bm2.to_mesh(top_mesh)
    bm2.free()

    top = bpy.data.objects.new("Arena_Top", top_mesh)
    bpy.context.collection.objects.link(top)
    top.location = (0, 0, ring_top_z + 0.01)

    top_mat = _material_arena_block_yellow(
        "mat_arena_top",
        color=arena_color,
        roughness=arena_roughness,
        specular=arena_specular,
    )
    top.data.materials.append(top_mat)
    _shade_auto_smooth_mesh(top, 88.0)
    _arena_prism_bevel(top, 0.022, angle_deg=55.0)

    # Floor plane (replaces MuJoCo outside_floor to avoid z-fighting)
    bpy.ops.mesh.primitive_plane_add(size=500, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Arena_Floor"
    floor_mat = _material_arena_floor_textured("mat_arena_floor")
    floor.data.materials.append(floor_mat)
    _shade_auto_smooth_mesh(floor, 89.0)


def _add_ring_edge_glow(ring_top_z: float = 2.0, ring_radius: float = 7.5):
    """Add a subtle neutral rim highlight at the platform edge."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=ring_radius,
        minor_radius=0.03,
        major_segments=64,
        minor_segments=8,
        location=(0, 0, ring_top_z + 0.005),
    )
    glow = bpy.context.active_object
    glow.name = "RingEdgeGlow"

    mat = bpy.data.materials.new(name="mat_ring_glow")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        gr, gg, gb = 0.62, 0.62, 0.63
        bsdf.inputs["Base Color"].default_value = (gr, gg, gb, 1.0)
        bsdf.inputs["Emission Color"].default_value = (gr, gg, gb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.85
        bsdf.inputs["Roughness"].default_value = 0.0
    glow.data.materials.append(mat)
    _shade_auto_smooth_mesh(glow, 52.0)


# ═══════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════

def _geom_xy_half_extent(gm: dict) -> float:
    """Rough horizontal half-extent for framing (MuJoCo geom size convention)."""
    typ = gm.get("type", "box")
    sz = gm.get("size") or [0.45, 0.45, 0.45]
    try:
        if typ == "box":
            return float(max(sz[0], sz[1]))
        if typ == "cylinder":
            return float(sz[0])
        if typ == "capsule":
            return float(sz[0]) + float(sz[1])
        if typ == "sphere":
            return float(sz[0])
        if typ == "mesh":
            return float(max(sz[0], sz[1])) if len(sz) >= 2 else 0.55
    except (TypeError, ValueError, IndexError):
        pass
    return 0.55


def setup_mujoco_tracking_camera(
    scene: bpy.types.Scene,
    bot_red_xy: tuple[float, float, float],
    bot_blue_xy: tuple[float, float, float],
    *,
    ring_top_z: float = 2.0,
    elevation_deg: float = -30.0,
    azimuth_deg: float = 90.0,
    min_distance: float = 6.0,
    padding_factor: float = 2.0,
    lens_mm: float = 35.0,
):
    """Static-frame camera that mirrors `episode.py:_make_tracking_camera`.

    Designed for keyframe stills: places the camera at the same offset MuJoCo's
    tracking camera uses (azimuth/elevation/distance from the bot midpoint),
    with `distance = max(min_distance, separation * padding_factor)`. No intro
    pull-back, no animation.
    """
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "TrackingCamera_Static"
    scene.camera = cam
    cam.data.lens = float(lens_mm)
    cam.data.clip_end = 500

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, ring_top_z))
    target = bpy.context.active_object
    target.name = "TrackingCamera_Target"

    constraint = cam.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    red = np.asarray(bot_red_xy, dtype=np.float64)
    blue = np.asarray(bot_blue_xy, dtype=np.float64)
    midpoint = (red + blue) * 0.5
    separation = float(np.linalg.norm(red[:2] - blue[:2]))
    distance = max(min_distance, separation * padding_factor)

    elev = math.radians(elevation_deg)  # negative ⇒ camera above lookat
    az = math.radians(azimuth_deg)

    tgt_x = float(midpoint[0])
    tgt_y = float(midpoint[1])
    tgt_z = ring_top_z

    # Same trig as the cinematic path: sin(-elev) → +z when elev is negative.
    x = tgt_x + distance * math.cos(elev) * math.cos(az)
    y = tgt_y + distance * math.cos(elev) * math.sin(az)
    z = tgt_z + distance * math.sin(-elev)

    cam.location = (x, y, z)
    target.location = (tgt_x, tgt_y, tgt_z)
    return cam, target


def _setup_camera(
    scene: bpy.types.Scene,
    n_frames: int,
    fps: float,
    geom_xpos: np.ndarray,
    geom_meta: list[dict],
    cfg: dict,
    ring_top_z: float = 2.0,
    frame_step: int = 3,
    bot_xpos: np.ndarray | None = None,
):
    """Wide arena establishing shot, then fixed-azimuth orbit around bot midpoint.

    When ``bot_xpos`` is provided (shape n_frames x 2 x 3, red then blue root body
    origins), the track target uses that midpoint; otherwise chassis geom centers.
    """
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "CinematicCamera"
    scene.camera = cam
    cam.data.lens = 35
    cam.data.clip_end = 500

    # Depth of field
    if cfg.get("depth_of_field"):
        cam.data.dof.use_dof = True
        cam.data.dof.aperture_fstop = 8.0

    # Track-to constraint
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, ring_top_z))
    target = bpy.context.active_object
    target.name = "CameraTarget"

    constraint = cam.constraints.new('TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # DOF focus target
    if cfg.get("depth_of_field"):
        cam.data.dof.focus_object = target

    red_idx, blue_idx = _red_blue_chassis_indices(geom_meta)

    red_gm = geom_meta[red_idx] if red_idx < len(geom_meta) else {}
    blue_gm = geom_meta[blue_idx] if blue_idx < len(geom_meta) else {}
    half_w_red = _geom_xy_half_extent(red_gm)
    half_w_blue = _geom_xy_half_extent(blue_gm)

    total_src_frames = geom_xpos.shape[0]
    elevation = math.radians(-34)
    # Minimum distance when bots pile up. Lowered from 9.0 → 7.2 to bring the
    # camera closer in tight contact frames; the dynamic `required_distance`
    # below still backs off as bots separate so both stay in frame.
    dist_floor = 7.2
    alpha_zoom = 0.28  # follow-phase distance smoothing (midpoint is never lagged)
    alpha_zoom_blend = 0.12  # gentler pull during wide→close blend
    # Smaller assumed half-FOV ⇒ computed stand-off is farther (more headroom in frame).
    half_fov_h = math.radians(27)
    # Tighter framing buffer (1.85 → 1.20). Together with the lower
    # dist_floor this lands the camera ~25–30% closer at typical separations
    # while still guaranteeing both bots fit (the +half_w_red+half_w_blue
    # terms upstream keep the sides safe).
    frame_padding = 1.20
    # No extra global pull-back on follow; intro stays at its own intro_pull.
    distance_scale = 1.00
    intro_pull = 1.06
    # Fixed camera heading in XY (no orbit / broadside tracking).
    fixed_azimuth = math.radians(35)
    # Establishing shot: ring center + same azimuth + stand-off to fit whole octagon.
    intro_len = max(10, int(n_frames * 0.14))
    # Longer blend + slower distance ramp (see u_dist below) = slower wide→close.
    blend_len = min(52, max(22, int(n_frames * 0.20)))
    mz_ring = ring_top_z * 0.52

    smooth_distance = None

    cos_elev = math.cos(elevation)
    tan_h = math.tan(half_fov_h)
    denom = max(1e-6, tan_h * abs(cos_elev))
    # Horizontal extent from ring center to far octagon wall (~7.5 m top R) + margin.
    arena_half_extent_xy = 8.45
    d_intro = max(22.0, (arena_half_extent_xy + 1.15) / denom * intro_pull)

    for render_frame, src_frame in enumerate(range(0, total_src_frames, frame_step)):
        blender_frame = render_frame + 1

        if (
            bot_xpos is not None
            and src_frame < int(bot_xpos.shape[0])
            and int(bot_xpos.shape[1]) >= 2
        ):
            red_pos = np.asarray(bot_xpos[src_frame, 0], dtype=np.float64)
            blue_pos = np.asarray(bot_xpos[src_frame, 1], dtype=np.float64)
        else:
            red_pos = geom_xpos[src_frame, red_idx]
            blue_pos = geom_xpos[src_frame, blue_idx]
        midpoint = (red_pos + blue_pos) * 0.5
        separation = float(np.linalg.norm(red_pos[:2] - blue_pos[:2]))

        # Horizontal span slack (conservative vs separation; camera azimuth is fixed).
        half_span = separation * 0.5 + half_w_red + half_w_blue + frame_padding
        required_distance = half_span / denom
        raw_follow_dist = max(dist_floor, (required_distance + 0.95) * distance_scale)

        mz = float(midpoint[2])
        mz = max(0.25, min(mz, ring_top_z + 6.0))
        cx = float(midpoint[0])
        cy = float(midpoint[1])

        if render_frame < intro_len:
            tgt_x, tgt_y, tgt_z = 0.0, 0.0, mz_ring
            if smooth_distance is None:
                smooth_distance = d_intro
            else:
                smooth_distance = 0.16 * d_intro + 0.84 * smooth_distance
        elif render_frame < intro_len + blend_len:
            raw_u = (render_frame - intro_len + 1) / float(blend_len)
            raw_u = max(0.0, min(1.0, raw_u))
            # Target moves on smoothstep; zoom uses a slower power curve so we hold wide longer.
            u_tgt = raw_u * raw_u * (3.0 - 2.0 * raw_u)
            u_dist = raw_u ** 1.48
            tgt_x = cx * u_tgt
            tgt_y = cy * u_tgt
            tgt_z = mz * u_tgt + mz_ring * (1.0 - u_tgt)
            d_blend = (1.0 - u_dist) * d_intro + u_dist * raw_follow_dist
            if smooth_distance is None:
                smooth_distance = d_blend
            else:
                smooth_distance = (
                    alpha_zoom_blend * d_blend + (1.0 - alpha_zoom_blend) * smooth_distance
                )
        else:
            tgt_x, tgt_y, tgt_z = cx, cy, mz
            if smooth_distance is None:
                smooth_distance = raw_follow_dist
            else:
                smooth_distance = alpha_zoom * raw_follow_dist + (1.0 - alpha_zoom) * smooth_distance

        azimuth = fixed_azimuth

        x = tgt_x + smooth_distance * math.cos(elevation) * math.cos(azimuth)
        y = tgt_y + smooth_distance * math.cos(elevation) * math.sin(azimuth)
        z = tgt_z + smooth_distance * math.sin(-elevation)

        cam.location = (x, y, z)
        cam.keyframe_insert(data_path="location", frame=blender_frame)

        target.location = (tgt_x, tgt_y, tgt_z)
        target.keyframe_insert(data_path="location", frame=blender_frame)

    # Linear interpolation for all keyframes
    for obj in (cam, target):
        try:
            if obj.animation_data and obj.animation_data.action:
                fcurves = getattr(obj.animation_data.action, 'fcurves', None)
                if fcurves is None:
                    for slot in getattr(obj.animation_data.action, 'slots', []):
                        for ch in getattr(slot, 'channels', []):
                            for fc in getattr(ch, 'fcurves', []):
                                for kp in fc.keyframe_points:
                                    kp.interpolation = 'LINEAR'
                else:
                    for fc in fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = 'LINEAR'
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Compositor (color grading, vignette, bloom)
# ═══════════════════════════════════════════════════════════════════

def _cfg_saturation_mult(cfg: dict) -> float:
    """Hue/Sat node: 1.0 = unchanged; >1 boosts chroma. PRESETS use bools: False → 1.0, True → DEFAULT numeric."""
    default_num = DEFAULT_CONFIG["saturation"]
    if not isinstance(default_num, (int, float)):
        default_num = 2.1
    v = cfg.get("saturation", default_num)
    if isinstance(v, bool):
        return float(default_num) if v else 1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default_num)


def _setup_compositor(scene: bpy.types.Scene, cfg: dict):
    """Set up compositing nodes for cinematic post-processing."""
    # Debug: bypass compositor entirely (no bloom/grade/vignette/soften).
    if os.environ.get("NO_POST"):
        try:
            scene.use_nodes = False
        except Exception:
            pass
        try:
            if hasattr(scene, "compositing_node_group"):
                scene.compositing_node_group = None
        except Exception:
            pass
        return
    b5 = bpy.app.version >= (5, 0, 0)
    # Blender 5+: compositor tree is compositing_node_group + NodeGroupOutput (not node_tree / Composite).
    tree = None
    if b5 and hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
    if tree is None:
        tree = getattr(scene, "node_tree", None)
    if tree is None:
        try:
            tree = bpy.data.node_groups.new("MjArenaCompositor", "CompositorNodeTree")
            if b5 and hasattr(scene, "compositing_node_group"):
                scene.compositing_node_group = tree
            else:
                try:
                    scene.use_nodes = True
                except Exception:
                    pass
                scene.node_tree = tree
        except Exception:
            return
    if tree is None:
        return
    if not b5:
        try:
            scene.use_nodes = True
        except Exception:
            pass
    nodes = tree.nodes
    links = tree.links

    # Clear defaults
    for node in nodes:
        nodes.remove(node)

    # Render layers input
    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    current_output = rl.outputs["Image"]
    x_offset = 300

    # Bloom / Glare
    if cfg.get("bloom"):
        glare = nodes.new("CompositorNodeGlare")
        glare.glare_type = 'FOG_GLOW'
        glare.quality = 'HIGH'
        glare.mix = -0.95  # subtle
        glare.threshold = 0.8
        glare.location = (x_offset, 0)
        links.new(current_output, glare.inputs["Image"])
        current_output = glare.outputs["Image"]
        x_offset += 300

    # Color grading (warm highlights, cool shadows)
    if cfg.get("color_grading"):
        cb = nodes.new("CompositorNodeColorBalance")
        cb.correction_method = 'LIFT_GAMMA_GAIN'
        cb.lift = (1.0, 1.0, 1.0)
        cb.gamma = (1.0, 1.0, 1.0)
        cb.gain = (1.0, 1.0, 1.0)
        cb.location = (x_offset, 0)
        links.new(current_output, cb.inputs["Image"])
        current_output = cb.outputs["Image"]
        x_offset += 300

    # Vignette
    if cfg.get("vignette"):
        ellipse = nodes.new("CompositorNodeEllipseMask")
        ellipse.width = 0.85
        ellipse.height = 0.85
        ellipse.location = (x_offset, -200)

        blur = nodes.new("CompositorNodeBlur")
        blur.size_x = 200
        blur.size_y = 200
        blur.location = (x_offset + 200, -200)
        links.new(ellipse.outputs["Mask"], blur.inputs["Image"])

        mix = nodes.new("CompositorNodeMixRGB")
        mix.blend_type = 'MULTIPLY'
        mix.inputs["Fac"].default_value = 0.3  # subtle
        mix.location = (x_offset + 400, 0)
        links.new(current_output, mix.inputs[1])
        links.new(blur.outputs["Image"], mix.inputs[2])
        current_output = mix.outputs["Image"]
        x_offset += 600

    # Saturation boost (post) — keeps materials untouched.
    sat = nodes.new("CompositorNodeHueSat")
    sat.location = (x_offset, 0)
    sat.inputs["Hue"].default_value = 0.5
    sat_mult = max(0.0, min(5.0, _cfg_saturation_mult(cfg)))
    # Blender 4.2+ renamed sockets Sat→Saturation, Val→Value.
    _sat_sock = sat.inputs.get("Saturation") or sat.inputs.get("Sat")
    _val_sock = sat.inputs.get("Value") or sat.inputs.get("Val")
    if _sat_sock is not None:
        _sat_sock.default_value = sat_mult
    if _val_sock is not None:
        _val_sock.default_value = 1.0
    sat.inputs["Fac"].default_value = 1.0
    links.new(current_output, sat.inputs["Image"])
    current_output = sat.outputs["Image"]
    x_offset += 260

    # Mild global contrast reduction (optional).
    if cfg.get("soften_contrast", True):
        soften = nodes.new("CompositorNodeBrightContrast")
        soften.location = (x_offset, 0)
        soften.inputs["Bright"].default_value = 0.01
        soften.inputs["Contrast"].default_value = -0.06
        links.new(current_output, soften.inputs["Image"])
        current_output = soften.outputs["Image"]
        x_offset += 220

    # Output — Blender 5+ uses group output + tree interface socket; older uses Composite.
    if b5:
        comp_out = nodes.new("NodeGroupOutput")
        comp_out.location = (x_offset, 0)
        try:
            tree.interface.new_socket(
                name="Image",
                in_out="OUTPUT",
                socket_type="NodeSocketColor",
            )
        except RuntimeError:
            pass
        try:
            links.new(current_output, comp_out.inputs["Image"])
        except (KeyError, TypeError):
            links.new(current_output, comp_out.inputs[0])
    else:
        comp_out = nodes.new("CompositorNodeComposite")
        comp_out.location = (x_offset, 0)
        links.new(current_output, comp_out.inputs["Image"])


# ═══════════════════════════════════════════════════════════════════
# Scoreboard overlay
# ═══════════════════════════════════════════════════════════════════

def _add_scoreboard_overlay(output_dir: Path, metadata: dict, width: int, height: int):
    """Burn scoreboard text onto rendered frames using PIL (post-render).

    Much more reliable than Blender text objects which don't work well
    in camera-space across versions.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available — skipping scoreboard overlay")
        return

    red_name = str(metadata.get("red_name", "Red"))
    blue_name = str(metadata.get("blue_name", "Blue"))
    season = str(metadata.get("season", ""))
    tournament = str(metadata.get("tournament", ""))
    seed = str(metadata.get("seed", "0"))

    center_text = ""
    if season:
        center_text += season
    if tournament:
        center_text += f"  /  {tournament}"
    center_text += f"  /  Seed {seed}"
    center_text = center_text.strip("  /  ")

    # Top overlay area (no bar, text only)
    overlay_y = int(height * 0.04)
    font_size = int(height * 0.028)
    small_font_size = int(height * 0.02)

    # Load IBM Plex Mono (matches BuildArena website mono/code surfaces)
    font_dir = Path(__file__).parent / "fonts"
    try:
        font = ImageFont.truetype(str(font_dir / "IBMPlexMono-SemiBold.ttf"), font_size)
        small_font = ImageFont.truetype(str(font_dir / "IBMPlexMono-Regular.ttf"), small_font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", small_font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font

    frames = sorted(output_dir.glob("frame_*.png"))
    print(f"Adding scoreboard to {len(frames)} frames...")

    for frame_path in frames:
        img = Image.open(frame_path)
        draw = ImageDraw.Draw(img, 'RGBA')

        # Soft dark text shadow for legibility against bright backgrounds
        def _text_with_shadow(xy, text, fill, fnt):
            x, y = xy
            # Shadow
            draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 140), font=fnt)
            draw.text(xy, text, fill=fill, font=fnt)

        dot_r = int(font_size * 0.28)
        dot_y = overlay_y + font_size // 2

        # Red bot (left)
        x_left = int(width * 0.05)
        draw.ellipse([(x_left, dot_y - dot_r), (x_left + dot_r * 2, dot_y + dot_r)],
                     fill=(220, 70, 70, 255))
        _text_with_shadow((x_left + dot_r * 3, overlay_y), red_name,
                          (245, 245, 245, 255), font)

        # Blue bot (right)
        bbox = draw.textbbox((0, 0), blue_name, font=font)
        text_w = bbox[2] - bbox[0]
        x_right_text = int(width * 0.95) - text_w
        _text_with_shadow((x_right_text, overlay_y), blue_name,
                          (245, 245, 245, 255), font)
        draw.ellipse([(x_right_text - dot_r * 3, dot_y - dot_r),
                      (x_right_text - dot_r, dot_y + dot_r)],
                     fill=(70, 120, 220, 255))

        # Center info (smaller, dimmer)
        bbox = draw.textbbox((0, 0), center_text, font=small_font)
        center_w = bbox[2] - bbox[0]
        center_y = overlay_y + (font_size - small_font_size) // 2
        _text_with_shadow(((width - center_w) // 2, center_y), center_text,
                          (200, 200, 205, 230), small_font)

        img.save(frame_path)

    print("Scoreboard overlay complete")


# ═══════════════════════════════════════════════════════════════════
# Main build & render
# ═══════════════════════════════════════════════════════════════════

def build_and_render(
    trajectory_npz: Path,
    output_dir: Path,
    width: int = 1920,
    height: int = 1080,
    samples: int = 16,
    engine: str = "eevee",
    frame_step: int = 3,
    use_gpu: bool = True,
    cfg: dict | None = None,
):
    """Build Blender scene from trajectory data and render."""
    if cfg is None:
        cfg = dict(DEFAULT_CONFIG)

    # Load trajectory
    data = np.load(trajectory_npz, allow_pickle=True)
    geom_xpos = data["geom_xpos"]
    geom_xmat = data["geom_xmat"]
    geom_meta = json.loads(str(data["geom_meta"]))
    bot_xpos = data["bot_xpos"] if "bot_xpos" in data.files else None
    n_frames = int(data["n_frames"])
    n_geoms = int(data["n_geoms"])
    fps = float(data["fps"])

    # Match metadata for scoreboard
    metadata = {}
    for key in ("red_name", "blue_name", "season", "tournament", "seed", "winner", "winner_step"):
        if key in data:
            metadata[key] = str(data[key]) if isinstance(data[key], np.ndarray) else data[key]

    render_fps = int(fps / frame_step)
    render_n_frames = (n_frames + frame_step - 1) // frame_step

    print(f"Building scene: {n_frames} frames → {render_n_frames} render frames at {render_fps} fps")
    print(f"Engine: {engine}, samples: {samples}, resolution: {width}x{height}")
    print(f"Config: {json.dumps({k: v for k, v in cfg.items() if v}, indent=2)}")

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = render_n_frames
    scene.render.fps = render_fps
    try:
        scene.view_settings.exposure = 0.34
    except (AttributeError, TypeError):
        pass
    # Debug: neutralize tonemapping/look to isolate "why is yellow dim?"
    if os.environ.get("NEUTRAL_VIEW"):
        try:
            scene.view_settings.view_transform = "Standard"
        except Exception:
            pass
        try:
            scene.view_settings.look = "None"
        except Exception:
            pass
        try:
            scene.view_settings.exposure = 0.0
        except Exception:
            pass

    # Render engine
    if engine == "cycles":
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        try:
            scene.cycles.glossy_bounces = 10
        except (AttributeError, TypeError):
            pass
        # CYCLES_DEVICE env var picks CPU vs GPU. Default 'GPU' (Mac Metal).
        # Set 'CPU' on Linux clusters where CPU jobs are far easier to
        # schedule than GPU jobs — Cycles+OIDN denoiser still produces
        # excellent quality, just at higher per-render wall time.
        cycles_device = os.environ.get("CYCLES_DEVICE", "GPU").upper()
        if cycles_device == "CPU":
            scene.cycles.device = 'CPU'
        elif use_gpu:
            prefs = bpy.context.preferences.addons.get('cycles')
            if prefs:
                # Pick the right GPU backend per platform — Metal on macOS,
                # CUDA/OPTIX where available on Linux.
                backend = os.environ.get("CYCLES_GPU_BACKEND", "METAL")
                prefs.preferences.compute_device_type = backend
                scene.cycles.device = 'GPU'
    else:
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except TypeError:
            scene.render.engine = 'BLENDER_EEVEE'
        try:
            scene.eevee.taa_render_samples = samples
            scene.eevee.use_shadows = True
        except AttributeError:
            pass
        # Shadow noise reduction. AREA lights with soft penumbras speckle on
        # the floor when shadows are undersampled. These attrs only exist on
        # eevee-next (Blender 4.2+), guarded individually.
        try:
            shadow_rays = int(os.environ.get("SHADOW_RAY_COUNT", "2"))
        except ValueError:
            shadow_rays = 2
        for attr, value in (
            ("shadow_ray_count", shadow_rays), # 1–4, default 1; multiplies shadow samples per TAA tick
            ("shadow_step_count", 4),          # 1–16, volumetric/contact step count
            ("shadow_pool_size", "256MB"),     # bigger cache → fewer eviction-driven re-bakes
            # False ⇒ cleaner/harder shadow edges (slightly noisier penumbra → disable if banding)
            ("use_shadow_jittered_viewport", False),
        ):
            try:
                setattr(scene.eevee, attr, value)
            except (AttributeError, TypeError):
                pass
        # Metals need visible env + SSR (Eevee) / glossy rays (Cycles) to show reflections.
        for attr, value in (("use_ssr", True), ("use_screen_space_reflections", True)):
            try:
                setattr(scene.eevee, attr, value)
            except (AttributeError, TypeError):
                pass
        try:
            scene.eevee.use_gtao = True
        except (AttributeError, TypeError):
            pass

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    # Build scene
    _setup_world(scene, cfg)
    _setup_lighting(scene)
    _setup_camera(
        scene,
        render_n_frames,
        render_fps,
        geom_xpos,
        geom_meta,
        cfg,
        frame_step=frame_step,
        bot_xpos=bot_xpos,
    )

    # Build arena from clean Blender primitives (avoids MuJoCo mesh artifacts).
    # When KEEP_MJ_RING is set, the MuJoCo sumo_ring is rendered instead
    # (physics-faithful straight cylinder) and only a ground plane is synthesized.
    if os.environ.get("KEEP_MJ_RING"):
        bpy.ops.mesh.primitive_plane_add(size=500, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.name = "Arena_Floor"
        floor_mat = _material_arena_floor_textured("mat_arena_floor_mjring")
        floor.data.materials.append(floor_mat)
        _shade_auto_smooth_mesh(floor, 89.0)
    else:
        _setup_arena()

    # Ring edge glow
    if cfg.get("ring_edge_glow"):
        _add_ring_edge_glow()

    # Compositor: optional effects + mild global contrast softening (always).
    try:
        _setup_compositor(scene, cfg)
        # Without this, Blender may skip the compositor tree — Hue/Sat would have no effect on PNGs.
        try:
            scene.render.use_compositing = True
        except (AttributeError, TypeError):
            pass
        print(
            f"Compositor: use_nodes={getattr(scene, 'use_nodes', '?')}, "
            f"use_compositing={getattr(scene.render, 'use_compositing', '?')}, "
            f"saturation_mult={_cfg_saturation_mult(cfg)}",
            flush=True,
        )
    except Exception as e:
        print(f"Compositor setup skipped: {e}")

    # Create geom objects
    objects: list[bpy.types.Object | None] = []
    for gm in geom_meta:
        obj = _create_geom_object(gm, cfg)
        if obj is not None:
            mat = _create_material(gm["name"], gm["rgba"], gm.get("material", ""), cfg)
            obj.data.materials.append(mat)
            if obj.type == "MESH":
                _shade_auto_smooth_mesh(obj, 50.0)
        objects.append(obj)

    marker_red = marker_blue = None
    red_idx_mk = blue_idx_mk = 0
    red_z_above = blue_z_above = 0.35
    marker_z_pad = 0.52
    if cfg.get("bot_team_markers", True):
        red_idx_mk, blue_idx_mk = _red_blue_chassis_indices(geom_meta)
        rgm = geom_meta[red_idx_mk] if red_idx_mk < len(geom_meta) else {}
        bgm = geom_meta[blue_idx_mk] if blue_idx_mk < len(geom_meta) else {}
        red_z_above = _geom_z_top_half(rgm)
        blue_z_above = _geom_z_top_half(bgm)
        marker_z_pad = float(cfg.get("bot_marker_z_clearance", 0.52))
        mr = float(cfg.get("bot_marker_radius", 0.085))
        mz = float(cfg.get("bot_marker_flatness", 0.34))
        marker_red = _create_bot_team_marker(
            "BotTeamMarker_Red", _team_indicator_rgba(True), radius=mr, z_scale=mz
        )
        marker_blue = _create_bot_team_marker(
            "BotTeamMarker_Blue", _team_indicator_rgba(False), radius=mr, z_scale=mz
        )

    print(f"Created {sum(1 for o in objects if o is not None)} objects (skipped {sum(1 for o in objects if o is None)})")

    # Scoreboard is applied post-render (see after bpy.ops.render.render)

    # Compute frame mapping (with slow-motion near winner_step if enabled)
    frame_map = []  # list of src_frame indices to render
    if cfg.get("slow_motion") and metadata.get("winner_step"):
        winner_step = int(metadata["winner_step"])
        slow_window = 100  # frames before winner_step to slow down
        slow_start = max(0, winner_step - slow_window)
        for src in range(0, n_frames, frame_step):
            if slow_start <= src <= winner_step:
                # Add intermediate frames for slow-mo (half step)
                frame_map.append(src)
                if src + frame_step // 2 < n_frames and src + frame_step // 2 <= winner_step:
                    frame_map.append(src + frame_step // 2)
            else:
                frame_map.append(src)
    else:
        frame_map = list(range(0, n_frames, frame_step))

    # Single-frame iteration mode: if RENDER_ONLY_FRAME env var is set,
    # clamp the render to that one output frame (1-indexed over the computed frame_map).
    only_frame_env = os.environ.get("RENDER_ONLY_FRAME")
    if only_frame_env:
        try:
            idx = max(1, min(int(only_frame_env), len(frame_map))) - 1
            frame_map = [frame_map[idx]]
            print(f"RENDER_ONLY_FRAME={only_frame_env}: rendering 1 frame (src {frame_map[0]})")
        except ValueError:
            print(f"RENDER_ONLY_FRAME='{only_frame_env}' is not an int — ignoring")

    # Update render frame count if slow-mo or single-frame changed it
    if len(frame_map) != render_n_frames:
        render_n_frames = len(frame_map)
        scene.frame_end = render_n_frames
        print(f"Render frame count now {render_n_frames}")

    # Keyframe transforms
    print(f"Keyframing {render_n_frames} frames...")
    for render_frame, src_frame in enumerate(frame_map):
        blender_frame = render_frame + 1
        for gid, obj in enumerate(objects):
            if obj is None:
                continue
            pos = geom_xpos[src_frame, gid]
            xmat = geom_xmat[src_frame, gid]
            z_offset = 0.0
            obj.location = (float(pos[0]), float(pos[1]), float(pos[2]) + z_offset)
            obj.rotation_euler = _xmat_to_euler(xmat)
            obj.keyframe_insert(data_path="location", frame=blender_frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=blender_frame)

        if marker_red is not None and marker_blue is not None:
            if (
                bot_xpos is not None
                and src_frame < int(bot_xpos.shape[0])
                and int(bot_xpos.shape[1]) >= 2
            ):
                rp = np.asarray(bot_xpos[src_frame, 0], dtype=np.float64)
                bp = np.asarray(bot_xpos[src_frame, 1], dtype=np.float64)
            else:
                rp = geom_xpos[src_frame, red_idx_mk]
                bp = geom_xpos[src_frame, blue_idx_mk]
            marker_red.location = (
                float(rp[0]),
                float(rp[1]),
                float(rp[2]) + red_z_above + marker_z_pad,
            )
            marker_blue.location = (
                float(bp[0]),
                float(bp[1]),
                float(bp[2]) + blue_z_above + marker_z_pad,
            )
            # World-locked orientation (never rolls/pitches with the robot).
            marker_red.rotation_euler = (0.0, 0.0, 0.0)
            marker_blue.rotation_euler = (0.0, 0.0, 0.0)
            marker_red.keyframe_insert(data_path="location", frame=blender_frame)
            marker_red.keyframe_insert(data_path="rotation_euler", frame=blender_frame)
            marker_blue.keyframe_insert(data_path="location", frame=blender_frame)
            marker_blue.keyframe_insert(data_path="rotation_euler", frame=blender_frame)

    # Linear interpolation
    _objs_for_fcurves = [o for o in objects if o is not None]
    if marker_red is not None:
        _objs_for_fcurves.extend([marker_red, marker_blue])
    for obj in _objs_for_fcurves:
        if obj.animation_data is None:
            continue
        try:
            action = obj.animation_data.action
            fcurves = getattr(action, 'fcurves', None)
            if fcurves is None:
                for slot in getattr(action, 'slots', []):
                    for ch in getattr(slot, 'channels', []):
                        for fc in getattr(ch, 'fcurves', []):
                            for kp in fc.keyframe_points:
                                kp.interpolation = 'LINEAR'
            else:
                for fc in fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
        except Exception:
            pass

    # Render
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_dir / "frame_")

    # Estimate render time
    est_per_frame = 0.5 if engine == "eevee" and samples <= 16 else (2.0 if engine == "eevee" else 20.0)
    est_total = est_per_frame * render_n_frames
    # Check for existing frames to support --continue
    existing_frames = sorted(output_dir.glob("frame_*.png"))
    if existing_frames:
        last_frame = int(existing_frames[-1].stem.split("_")[1])
        if last_frame >= render_n_frames:
            print(f"All {render_n_frames} frames already rendered — skipping")
            est_total = 0
        else:
            scene.frame_start = last_frame + 1
            remaining = render_n_frames - last_frame
            est_total = est_per_frame * remaining
            print(f"Resuming from frame {last_frame + 1} ({last_frame} already done, {remaining} remaining)")

    print(f"Rendering frames {scene.frame_start}-{render_n_frames} at {width}x{height}, {samples} samples...")
    print(f"Estimated time: {est_total/60:.0f} min ({est_per_frame:.1f}s/frame)")
    if est_total > 0:
        try:
            scene.render.use_compositing = True
        except (AttributeError, TypeError):
            pass
        bpy.ops.render.render(animation=True)
    print(f"Frames saved to {output_dir}")

    # Scoreboard overlay is applied by render_cinematic.py (system Python has PIL)

    # Encode to video with ffmpeg
    import subprocess, shutil
    ffmpeg = shutil.which("ffmpeg")
    video_path = output_dir / "render.mp4"
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(render_fps),
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(video_path),
        ]
        print(f"Encoding video: {video_path}")
        subprocess.run(cmd, capture_output=True)
        print(f"Done! Video: {video_path}")
    else:
        print("ffmpeg not found — frames saved but no video encoded")


if __name__ == "__main__":
    if bpy is None:
        print("ERROR: This script must be run inside Blender's Python interpreter.")
        print("Usage: blender --background --python render_blender.py -- trajectory.npz output_dir ...")
        sys.exit(1)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 2:
        print("Usage: blender --background --python render_blender.py -- trajectory.npz output_dir [width] [height] [samples] [engine] [frame_step] [config_json]")
        sys.exit(1)

    trajectory_npz = Path(argv[0])
    output_dir = Path(argv[1])
    width = int(argv[2]) if len(argv) > 2 else 1920
    height = int(argv[3]) if len(argv) > 3 else 1080
    samples = int(argv[4]) if len(argv) > 4 else 16
    engine = argv[5] if len(argv) > 5 else "eevee"
    frame_step = int(argv[6]) if len(argv) > 6 else 3

    # Load config from JSON arg or use defaults
    cfg = dict(DEFAULT_CONFIG)
    if len(argv) > 7:
        try:
            cfg.update(json.loads(argv[7]))
        except (json.JSONDecodeError, ValueError):
            # Try loading from file
            p = Path(argv[7])
            if p.exists():
                cfg.update(json.loads(p.read_text()))

    build_and_render(trajectory_npz, output_dir, width, height, samples, engine, frame_step, cfg=cfg)