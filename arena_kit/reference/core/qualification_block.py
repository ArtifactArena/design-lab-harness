"""The qualification opponent, resolved from the run's material palette.

The baseline block is an arena asset like any other, so its mass, friction and
contact priority come from ``configs/rules/materials_store.yaml`` instead of being
hand-picked in the XML — the same step (:func:`apply_material_properties`) a robot
goes through before composition. The 3D block is palette plastic in a
1.2 x 1.2 x 0.25 m slab: 0.36 m3 x 950 kg/m3 = 342 kg (user decision 2026-09-16,
audit flag F-N11).

``mass=`` is additionally baked into the asset so that a raw compile of the file —
the viewer, an ad-hoc composition — weighs the same block as qualification does;
:mod:`tests.test_baseline_block` fails if that baked number ever drifts from the
palette product.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

from mjarena.design_shop.utils import apply_material_properties, load_constraints_from_yaml

ASSETS_DIR = Path(__file__).parent / "assets"


def block_asset_path(physics_mode: str) -> Path:
    """The baseline block asset for ``physics_mode`` ("3d" or "2d")."""
    name = "stationary_block_3d.xml" if physics_mode == "3d" else "stationary_block.xml"
    return ASSETS_DIR / name


def qualification_block_xml(constraints_path: Path, physics_mode: str = "3d") -> str:
    """Block XML with palette mass, sliding friction and contact priority applied.

    Args:
        constraints_path: the run's rules YAML; ``materials_store.yaml`` next to it
            supplies the palette.
        physics_mode: "3d" (the slab) or "2d" (the legacy planar block).

    Raises:
        KeyError: the rules directory carries no material palette.
        RuntimeError: a block geom names a material the palette does not define, or
            the palette step reports an error.
    """
    source = block_asset_path(physics_mode).read_text(encoding="utf-8")
    constraints: Dict[str, Any] = load_constraints_from_yaml(Path(constraints_path))
    palette: Dict[str, Any] = constraints["materials"]

    # A typo'd material would otherwise be skipped in silence and leave the block at
    # MuJoCo's default density.
    for geom in ET.fromstring(source).iter("geom"):
        material = geom.get("material")
        if material is not None and material not in palette:
            raise RuntimeError(
                f"{block_asset_path(physics_mode).name}: geom "
                f"'{geom.get('name', 'unnamed')}' uses material='{material}', "
                f"which is not in the palette ({sorted(palette)})."
            )

    xml, errors, _info = apply_material_properties(source, palette)
    if errors:
        raise RuntimeError(
            f"Material palette could not be applied to "
            f"{block_asset_path(physics_mode).name}: " + "; ".join(errors)
        )
    return xml


def write_qualification_block(
    out_path: Path, constraints_path: Path, physics_mode: str = "3d"
) -> Path:
    """Write :func:`qualification_block_xml` to ``out_path`` and return it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(qualification_block_xml(constraints_path, physics_mode), encoding="utf-8")
    return out_path
