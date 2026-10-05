"""Surface proximity without convexifying concave mesh/SDF geometry.

MuJoCo's SDF collider only reports nearby penetrations, so separated SDFs
are queried against their original triangles using convex distance queries.
"""
from itertools import combinations
import numpy as np
import mujoco


def _closest_simplex(points):
    """Closest point in a simplex, retaining its barycentric witnesses."""
    best = None
    for count in range(1, min(4, len(points)) + 1):
        for ids in combinations(range(len(points)), count):
            vertices = np.array([points[i][0] for i in ids])
            if count == 1:
                weights = np.ones(1)
            else:
                edges = (vertices[1:] - vertices[0]).T
                weights = np.linalg.lstsq(edges, -vertices[0], rcond=None)[0]
                weights = np.r_[1 - weights.sum(), weights]
            if np.min(weights) < -1e-9:
                continue
            weights = np.maximum(weights, 0); weights /= weights.sum()
            p = weights @ vertices
            norm = float(p @ p)
            if best is None or norm < best[0]:
                best = norm, ids, weights, p
    _, ids, weights, p = best
    active = [(points[i], w) for i, w in zip(ids, weights) if w > 1e-12]
    a = sum(w * point[1] for point, w in active)
    b = sum(w * point[2] for point, w in active)
    return [point for point, _ in active], p, a, b


def convex_distance(support_a, support_b, direction):
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1., 0., 0.])
    direction = direction / np.linalg.norm(direction)
    def vertex(v):
        a, b = support_a(v), support_b(-v)
        return a-b, a, b
    simplex = [vertex(direction)]
    for _ in range(256):
        simplex, p, a, b = _closest_simplex(simplex)
        distance = float(np.linalg.norm(p))
        if distance < 1e-10:
            return 0., a, b
        new = vertex(-p / distance)
        # The support plane gives a lower bound; the simplex witnesses give
        # an upper bound. Bound error in meters, including near touching.
        lower = max(0., float(p @ new[0]) / distance)
        if distance-lower <= 1e-9 * max(1., distance):
            return distance, a, b
        if any(np.linalg.norm(new[0]-v[0]) < 1e-12 for v in simplex):
            return float(np.linalg.norm(p)), a, b
        simplex.append(new)
    simplex, p, a, b = _closest_simplex(simplex)
    return float(np.linalg.norm(p)), a, b


def _triangle_support(vertices):
    return lambda direction: vertices[int(np.argmax(vertices @ direction))]


def _on_mesh_surface(point, triangles):
    """Certify that a hull witness also lies on an original mesh triangle."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    normal = np.cross(b-a, c-a)
    area_squared = np.einsum('ij,ij->i', normal, normal)
    valid = area_squared > 0
    plane = np.abs(np.einsum('ij,ij->i', point-a, normal))
    valid &= plane <= 1e-9 * np.sqrt(area_squared)
    if not np.any(valid):
        return False
    a, b, c, normal, area_squared = a[valid], b[valid], c[valid], normal[valid], area_squared[valid]
    alpha = np.einsum('ij,ij->i', np.cross(b-point, c-point), normal) / area_squared
    beta = np.einsum('ij,ij->i', np.cross(c-point, a-point), normal) / area_squared
    gamma = 1-alpha-beta
    return bool(np.any((alpha >= -1e-12) & (beta >= -1e-12) & (gamma >= -1e-12)))


def _inside_mesh(point, triangles):
    # Sum oriented solid angles of the closed, consistently wound surface.
    # Ray-hit parity misclassifies rays that only graze an edge/vertex; merely
    # deduplicating their hits still counts a tangency as a boundary crossing.
    vectors = triangles - point
    lengths = np.linalg.norm(vectors, axis=2)
    if np.any(lengths == 0):
        return True  # On the surface: separation is zero either way.
    a, b, c = np.moveaxis(vectors / lengths[:, :, None], 1, 0)
    numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = (1 + np.einsum('ij,ij->i', a, b)
                   + np.einsum('ij,ij->i', b, c)
                   + np.einsum('ij,ij->i', c, a))
    winding = np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)
    # Preserve odd/even shell containment, including empty nested cavities.
    return bool(int(np.rint(abs(winding))) % 2)


def primitive_support(kind, size, position, rotation):
    def support(direction):
        v = rotation.T @ direction
        norm = np.linalg.norm(v)
        unit = v / norm if norm > 1e-15 else np.array([1., 0., 0.])
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            p = size[0]*unit
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            p = size[0]*unit; p[2] += size[1]*(1 if v[2] >= 0 else -1)
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            xy = np.linalg.norm(v[:2])
            p = np.r_[size[0]*v[:2]/xy if xy > 1e-15 else [size[0], 0], size[1]*(1 if v[2] >= 0 else -1)]
        elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            denominator = np.linalg.norm(size*v)
            p = size*size*v/denominator if denominator > 1e-15 else np.array([size[0], 0., 0.])
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            p = size*np.where(v >= 0, 1., -1.)
        else:
            raise ValueError(f'Unsupported proximity geom type {kind}')
        return position + rotation @ p
    return support


class SurfaceQueries:
    def __init__(self, model):
        self.model = model
        self.types = np.asarray(model.geom_type).copy()
        self.sizes = np.asarray(model.geom_size).copy()
        self.meshes = {}
        self.vertices = {}
        self.components = {}
        for gid in range(model.ngeom):
            if self.types[gid] not in (int(mujoco.mjtGeom.mjGEOM_SDF), int(mujoco.mjtGeom.mjGEOM_MESH)):
                continue
            mid = int(model.geom_dataid[gid]); va = int(model.mesh_vertadr[mid]); vn = int(model.mesh_vertnum[mid])
            fa = int(model.mesh_faceadr[mid]); fn = int(model.mesh_facenum[mid])
            vertices = np.asarray(model.mesh_vert)[va:va+vn].copy()
            faces = np.asarray(model.mesh_face)[fa:fa+fn].copy()
            self.meshes[gid] = vertices[faces]
            self.vertices[gid] = vertices
            # One sample per connected shell is enough for containment tests
            # once surface intersections are also checked. Disconnected solids
            # can overlap even when their first component is entirely outside.
            parent = list(range(vn))
            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for face in faces:
                for index in face[1:]:
                    parent[find(int(index))] = find(int(face[0]))
            representatives = sorted({find(int(i)) for i in faces.flat})
            self.components[gid] = vertices[representatives]

    def update(self, data):
        self.positions = np.asarray(data.geom_xpos).copy()
        self.rotations = np.asarray(data.geom_xmat).reshape(-1, 3, 3).copy()
        self.triangles = {g: t @ self.rotations[g].T + self.positions[g] for g, t in self.meshes.items()}
        self.world_vertices = {g: v @ self.rotations[g].T + self.positions[g] for g, v in self.vertices.items()}
        self.component_points = {g: p @ self.rotations[g].T + self.positions[g] for g, p in self.components.items()}

    def distance(self, data, a, b):
        if a not in self.meshes and b not in self.meshes:
            # MuJoCo 3.10's collision-distance query can return zero for
            # separated primitives (including box/box). Query their convex
            # surfaces directly so proximity does not inherit that failure.
            sa = primitive_support(int(self.types[a]), self.sizes[a], self.positions[a], self.rotations[a])
            sb = primitive_support(int(self.types[b]), self.sizes[b], self.positions[b], self.rotations[b])
            return convex_distance(sa, sb, self.positions[b]-self.positions[a])
        ta, tb = self.triangles.get(a), self.triangles.get(b)
        sa = primitive_support(int(self.types[a]), self.sizes[a], self.positions[a], self.rotations[a]) if ta is None else None
        sb = primitive_support(int(self.types[b]), self.sizes[b], self.positions[b], self.rotations[b]) if tb is None else None
        # A convex-hull distance is a lower bound on the true mesh distance.
        # It is exact when BOTH witnesses lie on the original surfaces. This
        # certificate avoids expensive triangle searches without filling in
        # concavities or cavities; uncertified results use the full query below.
        hull_a = sa if ta is None else _triangle_support(self.world_vertices[a])
        hull_b = sb if tb is None else _triangle_support(self.world_vertices[b])
        hull = convex_distance(hull_a, hull_b, self.positions[b]-self.positions[a])
        direction = (hull[2]-hull[1])/hull[0] if hull[0] > 1e-8 else np.zeros(3)
        lower = float(direction @ (hull_b(-direction)-hull_a(direction)))
        if (hull[0] > 1e-8
                and hull[0]-lower <= 1e-9 * max(1., hull[0])
                and (ta is None or _on_mesh_surface(hull[1], ta))
                and (tb is None or _on_mesh_surface(hull[2], tb))):
            return hull
        # Handle complete containment, which surface-triangle distances alone miss.
        pa = self.positions[a] if ta is None else ta[0, 0]
        pb = self.positions[b] if tb is None else tb[0, 0]
        for point in self.component_points.get(b, [pb]):
            if ta is not None and _inside_mesh(point, ta):
                return 0., point.copy(), point.copy()
        for point in self.component_points.get(a, [pa]):
            if tb is not None and _inside_mesh(point, tb):
                return 0., point.copy(), point.copy()
        best = (float('inf'), pa, pb)
        # Sort conservative AABB lower bounds before expensive support queries.
        # Bounds only prune candidates that cannot improve the current result.
        pieces_a = list(ta) if ta is not None else [None]
        pieces_b = list(tb) if tb is not None else [None]
        def bounds(triangles, support):
            if triangles is not None:
                return triangles.min(axis=1), triangles.max(axis=1)
            axes = np.eye(3)
            lo = np.array([support(-axis)[i] for i, axis in enumerate(axes)])
            hi = np.array([support(axis)[i] for i, axis in enumerate(axes)])
            return lo[None, :], hi[None, :]
        amin, amax = bounds(ta, sa)
        bmin, bmax = bounds(tb, sb)
        a_lower = np.linalg.norm(np.maximum(np.maximum(amin-bmax.max(0), bmin.min(0)-amax), 0), axis=1)
        for i in np.argsort(a_lower, kind='stable'):
            if a_lower[i] >= best[0]:
                break
            tri_a = pieces_a[i]
            support_a = sa if tri_a is None else _triangle_support(tri_a)
            gaps = np.maximum(np.maximum(amin[i]-bmax, bmin-amax[i]), 0)
            lower = np.linalg.norm(gaps, axis=1)
            for j in np.argsort(lower, kind='stable'):
                if lower[j] >= best[0]:
                    break
                tri_b = pieces_b[j]
                support_b = sb if tri_b is None else _triangle_support(tri_b)
                result = convex_distance(support_a, support_b, self.positions[b]-self.positions[a])
                if result[0] < best[0]:
                    best = result
                    if best[0] <= 1e-8:
                        return best
        return best
