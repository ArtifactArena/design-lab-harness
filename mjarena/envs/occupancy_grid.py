"""2D occupancy grid builders for spatial observation.

Provides three grids built from existing obs fields:
- arena_grid: int8 occupancy (-1=outside, 0=free, 1=opponent, 2=self)
- arena_mass_grid: float32 mass-weighted occupancy
- edge_distance_grid: float32 signed distance to ring edge (static per match)

All grids share the same coordinate mapping:
    center = grid_size // 2
    world_x = (col - center) * cell_size
    world_y = (center - row) * cell_size
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GridCache:
    """Pre-computed grid data reused every timestep."""
    grid_size: int
    cell_size: float
    ring_radius: float
    # (N, N) float32: distance from each cell center to arena center
    dist_from_center: np.ndarray
    # (N, N) bool: True for cells inside the ring
    inside_mask: np.ndarray
    # (N, N) float32: static edge distance grid (ring_radius - dist)
    edge_distance_grid: np.ndarray
    # (N, N, 2) float32: world XY of each cell center
    cell_centers_xy: np.ndarray


def make_grid_cache(ring_radius: float, cell_size: float = 0.25) -> GridCache:
    """Pre-compute base template and cell center arrays. Called once at match init."""
    grid_size = int(2 * ring_radius / cell_size) + 1
    center = grid_size // 2

    # Cell center coordinates in world frame
    cols = np.arange(grid_size, dtype=np.float32)
    rows = np.arange(grid_size, dtype=np.float32)
    cx = (cols - center) * cell_size  # world x for each column
    cy = (center - rows) * cell_size  # world y for each row (y flipped)

    # Meshgrid: cell_x[r,c] = world x, cell_y[r,c] = world y
    cell_x, cell_y = np.meshgrid(cx, cy)
    cell_centers_xy = np.stack([cell_x, cell_y], axis=-1)  # (N, N, 2)

    dist_from_center = np.sqrt(cell_x ** 2 + cell_y ** 2)
    inside_mask = dist_from_center <= ring_radius
    edge_distance_grid = np.where(
        inside_mask,
        ring_radius - dist_from_center,
        -(dist_from_center - ring_radius),
    ).astype(np.float32)

    return GridCache(
        grid_size=grid_size,
        cell_size=cell_size,
        ring_radius=ring_radius,
        dist_from_center=dist_from_center,
        inside_mask=inside_mask,
        edge_distance_grid=edge_distance_grid,
        cell_centers_xy=cell_centers_xy,
    )


def _robot_mask(
    cache: GridCache,
    pos_xy: np.ndarray,
    bounding_radius: float,
) -> np.ndarray:
    """Return (N, N) bool mask of cells within bounding_radius of pos_xy."""
    dx = cache.cell_centers_xy[..., 0] - pos_xy[0]
    dy = cache.cell_centers_xy[..., 1] - pos_xy[1]
    return (dx ** 2 + dy ** 2) <= (bounding_radius ** 2)


def build_arena_grid(
    cache: GridCache,
    my_pos: np.ndarray,
    opp_pos: np.ndarray,
    my_bounding_radius: float,
    opp_bounding_radius: float,
) -> np.ndarray:
    """Build int8 (N,N) occupancy grid: -1=outside, 0=free, 1=opponent, 2=self."""
    grid = np.where(cache.inside_mask, np.int8(0), np.int8(-1))

    # Opponent first so self overwrites on overlap
    opp_mask = _robot_mask(cache, opp_pos[:2], opp_bounding_radius) & cache.inside_mask
    grid[opp_mask] = 1

    my_mask = _robot_mask(cache, my_pos[:2], my_bounding_radius) & cache.inside_mask
    grid[my_mask] = 2

    return grid


def build_arena_mass_grid(
    cache: GridCache,
    my_pos: np.ndarray,
    opp_pos: np.ndarray,
    my_bounding_radius: float,
    opp_bounding_radius: float,
    my_mass: float,
    opp_mass: float,
) -> np.ndarray:
    """Build float32 (N,N) mass grid: -1=outside, 0=free, mass value=robot."""
    grid = np.where(cache.inside_mask, np.float32(0.0), np.float32(-1.0))

    opp_mask = _robot_mask(cache, opp_pos[:2], opp_bounding_radius) & cache.inside_mask
    grid[opp_mask] = np.float32(opp_mass)

    my_mask = _robot_mask(cache, my_pos[:2], my_bounding_radius) & cache.inside_mask
    grid[my_mask] = np.float32(my_mass)

    return grid
