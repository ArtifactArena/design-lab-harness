import logging
import re
import numpy as np

from typing import Dict, List, Tuple, Optional

import mujoco
from lxml import etree


def enforce_contact_settings(model):
    """Apply arena contact rules even to previously processed robot XML."""
    model.geom_condim[:] = 6
    model.geom_friction[:, 1:] = [0.005, 0.0001]
    # Explicit contact pairs otherwise bypass geom contact parameters.
    for pair in range(model.npair):
        a, b = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        pa, pb = int(model.geom_priority[a]), int(model.geom_priority[b])
        fa, fb = float(model.geom_friction[a][0]), float(model.geom_friction[b][0])
        sliding = fa if pa > pb else fb if pb > pa else max(fa, fb)
        model.pair_dim[pair] = 6
        model.pair_friction[pair] = [sliding, sliding, 0.005, 0.0001, 0.0001]


def _etree(path: str) -> etree._ElementTree:
    with open(path, "rb") as f:
        return etree.parse(f)


def _find_spawn_and_ring(env_tree: etree._ElementTree) -> Tuple[np.ndarray, np.ndarray, float]:
    root = env_tree.getroot()
    # Find spawn/red and spawn/blue sites
    red_site = root.xpath(".//site[@name='spawn/left']")
    blue_site = root.xpath(".//site[@name='spawn/right']")
    if not red_site or not blue_site:
        raise ValueError("spawn/red or spawn/blue site missing in env XML.")
    def _pos(el):
        return np.fromstring(el.get("pos", "0 0 0"), sep=" ")
    red = _pos(red_site[0])[:3]
    blue = _pos(blue_site[0])[:3]
    # Find ring radius from cylinder size[0]
    ring_geom = root.xpath(".//geom[@name='sumo_ring']")
    if not ring_geom:
        raise ValueError("geom name='sumo_ring' missing in env XML.")
    size = np.fromstring(ring_geom[0].get("size"), sep=" ")
    radius = float(size[0])
    return red, blue, radius


def _collect_name_map(node: etree._Element, prefix: str) -> Dict[str, str]:
    """Map every name attr inside `node` to a prefixed version."""
    mapping: Dict[str, str] = {}
    for el in node.iter():
        if "name" in el.attrib:
            old = el.get("name")
            new = f"{prefix}{old}"
            mapping[old] = new
    return mapping


def _apply_name_map(root: etree._Element, mapping: Dict[str, str], ref_attrs: Optional[List[str]] = None):
    """Apply mapping to both `name` attrs and common reference attrs (joint, site, body, geom, sensor, tendon, actuator)."""
    if ref_attrs is None:
        # Common reference attributes across MJCF elements.
        # Include exclude/equality pair attributes like body1/body2, geom1/geom2 as well.
        ref_attrs = [
            "joint", "jointinparent", "site", "refsite", "cranksite", "slidersite", "body", "geom", "tendon", "actuator", "material", "texture", "mesh", "equality",
            "body1", "body2", "geom1", "geom2", "site1", "site2", "joint1", "joint2", "tendon1", "tendon2", "sidesite",
        ]
    for el in root.iter():
        # rename element's own name
        if "name" in el.attrib:
            n = el.get("name")
            if el.tag == "exclude":
                # Exclude names sometimes encode a pair like "A:B"; map parts individually
                if ":" in n:
                    n1, n2 = n.split(":", 1)
                    n1m = mapping.get(n1, n1)
                    n2m = mapping.get(n2, n2)
                    el.set("name", f"{n1m}:{n2m}")
                elif n in mapping:
                    el.set("name", mapping[n])
            else:
                if n in mapping:
                    el.set("name", mapping[n])
        # rename references in attributes
        for k, v in list(el.attrib.items()):
            if k in ref_attrs and v in mapping:
                el.set(k, mapping[v])


def _extract_robot_subtrees(robot_tree: etree._ElementTree) -> Dict[str, etree._Element]:
    """Return dict of top-level sections present in robot XML.

    If a tag appears multiple times (notably <asset>), merge its children
    into a single synthetic holder element so downstream composition can
    append all definitions.
    """
    root = robot_tree.getroot()
    d: Dict[str, etree._Element] = {}
    for tag in ["worldbody", "actuator", "sensor", "asset", "contact", "equality", "tendon"]:
        nodes = root.findall(tag)
        if not nodes:
            continue
        if len(nodes) > 1:
            holder = etree.Element(tag)
            for n in nodes:
                for child in list(n):
                    holder.append(child)
            d[tag] = holder
        else:
            d[tag] = nodes[0]
    return d


def _get_robot_root_body(worldbody: etree._Element) -> etree._Element:
    for child in worldbody:
        if child.tag == "body":
            return child
    raise ValueError("Robot XML has no <body> under <worldbody>.")


def _strip_team_prefixes(text: str) -> str:
    """Remove team prefixes (red_/blue_) so composition errors read in the model's
    own namespace. Fallback for raw MuJoCo errors that mention prefixed names the
    model never wrote (e.g. 'unknown transmission target red_axle_fl')."""
    return re.sub(r"\b(?:red_|blue_)", "", text)


def _validate_robot_composition(
    sections: Dict[str, etree._Element], root_body: etree._Element
) -> None:
    """Fail early with an actionable, model-namespace error when a robot is not a
    single connected kinematic tree.

    `compose_sumo_model` keeps only the FIRST top-level <body> under <worldbody>
    (via `_get_robot_root_body`) and silently discards any sibling top-level
    bodies. If actuators then reference joints/sites that lived in a discarded
    body, MuJoCo aborts with a cryptic 'unknown transmission target' error stated
    in the *prefixed* (red_/blue_) namespace — names the model never wrote and
    cannot act on. This detects that exact situation BEFORE prefixing (so names
    are the originals) and explains both the cause and the fix.

    No-op for a well-formed robot (single top-level body, all parts nested).
    """
    worldbody = sections.get("worldbody")
    if worldbody is None:
        return
    top_bodies = [c for c in worldbody if c.tag == "body"]
    dropped = top_bodies[1:] if len(top_bodies) > 1 else []
    if not dropped:
        return  # single-tree robot: composition cannot drop anything

    # Names that SURVIVE composition: those inside the kept root body subtree.
    kept_joints = {j.get("name") for j in root_body.iter("joint", "freejoint") if j.get("name")}
    kept_sites = {s.get("name") for s in root_body.iter("site") if s.get("name")}

    def _owner_body(el: etree._Element) -> str:
        p = el.getparent()
        while p is not None and p.tag != "body":
            p = p.getparent()
        return p.get("name", "<unnamed>") if p is not None else "<worldbody>"

    joint_owner = {
        j.get("name"): _owner_body(j)
        for j in worldbody.iter("joint", "freejoint")
        if j.get("name")
    }
    site_owner = {
        s.get("name"): _owner_body(s) for s in worldbody.iter("site") if s.get("name")
    }

    # The actual breakage: actuators referencing parts that won't survive.
    dangling: List[str] = []
    actuator = sections.get("actuator")
    if actuator is not None:
        for a in actuator.iter():
            jt = a.get("joint")
            if jt is not None and jt not in kept_joints:
                dangling.append(
                    f"'{a.get('name') or a.tag}' -> joint '{jt}' "
                    f"(defined in discarded body '{joint_owner.get(jt, '?')}')"
                )
            st = a.get("site")
            if st is not None and st not in kept_sites:
                dangling.append(
                    f"'{a.get('name') or a.tag}' -> site '{st}' "
                    f"(defined in discarded body '{site_owner.get(st, '?')}')"
                )

    # ANY discarded top-level body is a failure: the simulated robot would
    # silently differ from the designed one (a no-op warning would let the model
    # be scored on a mutilated bot). Always raise.
    root_name = root_body.get("name", "<root>")
    dropped_names = ", ".join(b.get("name", "<unnamed>") for b in dropped)
    msg = (
        "Robot assembly error: your robot is not a single connected body. The arena "
        f"uses the FIRST <body> under <worldbody> ('{root_name}') as the robot root "
        "and discards every other top-level <body>. "
        f"Discarded sibling bodies: {dropped_names}. "
    )
    if dangling:
        msg += (
            "After discarding them, these actuators reference parts that no longer "
            f"exist: {'; '.join(dangling)}. "
        )
    msg += (
        f"Fix: nest every moving body as a descendant of the root body '{root_name}' so "
        "the whole robot forms ONE kinematic tree. Only the root body may sit directly "
        "under <worldbody>, and it must carry the <freejoint>; e.g. place each wheel/limb "
        f"<body> INSIDE <body name=\"{root_name}\"> rather than as a sibling of it."
    )
    raise ValueError(msg)


def _recolor_robot_geoms(root: etree._Element, rgba: Tuple[float, float, float, float]) -> None:
    """Set RGBA on all geoms under the given robot root body.

    If a geom already has an rgba with a custom alpha, preserve that alpha
    while replacing the RGB components with the team color.
    """
    r, g, b, a = rgba
    for geom in root.iter("geom"):
        existing = geom.get("rgba")
        if existing is not None:
            vals = np.fromstring(existing, sep=" ")
            alpha = float(vals[3]) if vals.size >= 4 else a
            geom.set("rgba", f"{r} {g} {b} {alpha}")
        else:
            geom.set("rgba", f"{r} {g} {b} {a}")

def _mj_load(xml_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def clamp_actuator_gears(
    model: mujoco.MjModel,
    max_gear_to_inertia_ratio: float = 5000.0,
) -> List[str]:
    """Clamp actuator gear values based on the driven body's rotational inertia.

    For each actuator: gear ≤ min_body_inertia × max_gear_to_inertia_ratio.
    Uses the minimum diagonal element of the body's inertia tensor as a
    conservative estimate (for wheels, this IS the rotation-axis inertia).

    Prevents simulation instability from high torque on low-inertia bodies.
    Modifies model.actuator_gear in-place.

    Returns a list of human-readable messages for each clamped actuator.
    """
    log = logging.getLogger(__name__)
    clamped: List[str] = []
    for i in range(model.nu):
        # This inertia heuristic only applies to joint motors, not cables or
        # site mechanisms with configuration-dependent mechanical advantage.
        if int(model.actuator_trntype[i]) not in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            continue
        jnt_id = model.actuator_trnid[i, 0]
        body_id = model.jnt_bodyid[jnt_id]

        # Use minimum diagonal inertia (most conservative / easiest to spin)
        body_inertia_diag = model.body_inertia[body_id]  # [Ixx, Iyy, Izz]
        min_inertia = float(np.min(body_inertia_diag))

        gear_mag = float(np.linalg.norm(model.actuator_gear[i]))
        max_gear = min_inertia * max_gear_to_inertia_ratio

        if max_gear > 0 and gear_mag > max_gear:
            scale = max_gear / gear_mag
            model.actuator_gear[i] *= scale
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            msg = (
                f"Clamped {act_name}: gear {gear_mag:.0f} -> {max_gear:.0f} "
                f"(body '{body_name}' I_min={min_inertia:.4f} kg·m², "
                f"limit=I*{max_gear_to_inertia_ratio:.0f})"
            )
            log.warning(msg)
            clamped.append(msg)

    return clamped
