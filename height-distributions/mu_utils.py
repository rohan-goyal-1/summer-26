from scipy.optimize import minimize_scalar, brentq
import numpy as np


def height_distribution_2d(
    pos_p: np.ndarray,
    rad: np.ndarray,
    *,
    n_angles: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the angular height distribution for a 2D clump.

    Returns ``(angles, heights)`` where ``angles`` are uniformly spaced on
    ``[0, 2 pi)`` and ``heights[i]`` is the COM-to-perimeter distance along
    direction ``angles[i]`` (support function of the union of disks).
    """
    pos_p = np.asarray(pos_p, dtype=float)
    rad = np.asarray(rad, dtype=float)
    if pos_p.ndim != 2 or pos_p.shape[1] != 2:
        raise ValueError(f"pos_p must have shape (N, 2); got {pos_p.shape}.")
    if rad.shape != (pos_p.shape[0],):
        raise ValueError(
            f"rad must have shape ({pos_p.shape[0]},); got {rad.shape}."
        )
    if n_angles < 1:
        raise ValueError(f"n_angles must be >= 1; got {n_angles}.")

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    heights = np.max(pos_p @ dirs.T + rad[:, None], axis=0)
    return angles, heights


def mean_height_stats_2d(
    pos_p: np.ndarray,
    rad: np.ndarray,
    *,
    n_angles: int = 10_000,
) -> tuple[float, float, float, float]:
    """Mean height, mean squared height, and angular standard deviations.

    Height at polar angle ``theta`` is the distance from the particle center
    of mass to the perimeter along direction ``theta``: the support function
    of the union of disks with centers ``pos_p`` and radii ``rad``,

    ``h(theta) = max_i (pos_p_i . e_theta + rad_i)``,

    where ``pos_p`` is expressed in the COM frame. Also returns
    ``std(h(theta))`` and ``std(h(theta)^2)`` over uniformly sampled angles.
    """
    _, heights = height_distribution_2d(pos_p, rad, n_angles=n_angles)
    return mean_height_stats_from_distribution(heights)


def mean_height_stats_from_distribution(
    heights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Summary statistics for a precomputed angular height distribution."""
    heights = np.asarray(heights, dtype=float)
    heights_sq = heights**2
    return (
        float(np.mean(heights)),
        float(np.mean(heights_sq)),
        float(np.std(heights)),
        float(np.std(heights_sq)),
    )


def mean_edge_l_over_sigma_2d(
    pos_p: np.ndarray,
    rad: np.ndarray,
    *,
    n_surface: int | None = None,
    exclude_core: bool = True,
    core_tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Mean edge spacing-to-diameter ratio for a 2D GA clump.

    Surface asperities are ordered by polar angle around the clump COM.
    Each edge connects consecutive asperities (with wrap-around). For an
    edge between asperities ``i`` and ``j``:

    * spacing ``ell = |pos_p_i - pos_p_j|``
    * mean diameter ``sigma = (2 rad_i + 2 rad_j) / 2 = rad_i + rad_j``
    * ratio ``ell / sigma``

    When ``n_surface`` is given, only the first ``n_surface`` spheres are
    used (surface asperities in :func:`create_ga_state` solid-clump builds).
    Otherwise, when ``exclude_core`` is True, spheres at the origin are
    dropped before ordering.
    """
    pos_p = np.asarray(pos_p, dtype=float)
    rad = np.asarray(rad, dtype=float)
    if pos_p.ndim != 2 or pos_p.shape[1] != 2:
        raise ValueError(f"pos_p must have shape (N, 2); got {pos_p.shape}.")
    if rad.shape != (pos_p.shape[0],):
        raise ValueError(
            f"rad must have shape ({pos_p.shape[0]},); got {rad.shape}."
        )

    if n_surface is not None:
        if n_surface < 2 or n_surface > pos_p.shape[0]:
            raise ValueError(
                f"n_surface must satisfy 2 <= n_surface <= N; got {n_surface}."
            )
        pos_p = pos_p[:n_surface]
        rad = rad[:n_surface]
    elif exclude_core:
        mask = np.linalg.norm(pos_p, axis=1) > core_tol
        pos_p = pos_p[mask]
        rad = rad[mask]

    if pos_p.shape[0] < 2:
        raise ValueError("Need at least two surface asperities to define edges.")

    order = np.argsort(np.arctan2(pos_p[:, 1], pos_p[:, 0]))
    pos_p = pos_p[order]
    rad = rad[order]

    next_idx = np.arange(1, pos_p.shape[0] + 1) % pos_p.shape[0]
    spacing = np.linalg.norm(pos_p - pos_p[next_idx], axis=1)
    mean_diameter = rad + rad[next_idx]
    ratios = spacing / mean_diameter
    return float(np.mean(ratios)), ratios


def calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices):
    """
    Calculate the effective friction coefficient in 2D.
    """
    return 1 / np.sqrt(((2 * vertex_radius) / ((outer_radius - vertex_radius) * np.sin(np.pi / num_vertices))) ** 2 - 1)

def find_num_vertices_for_target_mu_eff_2d(
    target_mu_eff: float,
    vertex_radius: float,
    outer_radius: float,
    num_vertices_min: int = 3,
    num_vertices_max: int = 100):
    """
    Solve for the number of vertices that gives a desired effective friction coefficient in 2D.
    """
    best_nv = None
    best_mu = np.nan
    best_err = np.inf
    for nv in range(int(num_vertices_min), int(num_vertices_max) + 1):
        try:
            mu = float(calc_mu_eff_2d(vertex_radius, outer_radius, nv))
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError, TypeError):
            continue
        if not np.isfinite(mu):
            continue
        err = abs(mu - target_mu_eff)
        if err < best_err:
            best_nv, best_mu, best_err = nv, mu, err
    return best_nv, best_mu, best_err

def get_closest_vertex_radius_for_mu_eff_2d(mu_eff, outer_radius, num_vertices):
    """
    Solve for the vertex diameter that gives a desired effective friction coefficient in 2D.
    """
    # Calculate mathematically valid bounds
    sin_term = np.sin(np.pi / num_vertices)
    min_vertex_radius = outer_radius * sin_term / (2 + sin_term) + 1e-12
    max_vertex_radius = outer_radius - 1e-12
    
    # Check if target mu_eff is achievable
    max_mu_eff = calc_mu_eff_2d(min_vertex_radius, outer_radius, num_vertices)
    min_mu_eff = calc_mu_eff_2d(max_vertex_radius, outer_radius, num_vertices)
    
    if mu_eff > max_mu_eff or mu_eff < min_mu_eff:
        # Target mu_eff is outside achievable range
        return np.nan
    try:
        # Use root finding since we want calc_mu_eff(vertex_radius) = mu_eff
        def objective(vertex_radius):
            return calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices) - mu_eff
        
        # Brent's method is robust for this monotonic function
        result = brentq(objective, min_vertex_radius, max_vertex_radius, xtol=1e-12)
        return result
        
    except (ValueError, RuntimeError, ZeroDivisionError):
        # Fallback to bounded scalar minimization if root finding fails
        def obj_squared(vertex_radius):
            try:
                return (calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices) - mu_eff) ** 2
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.inf
        
        result = minimize_scalar(obj_squared, bounds=(min_vertex_radius, max_vertex_radius), method='bounded')
        return result.x if result.success else np.nan