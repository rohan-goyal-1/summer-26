import jax
jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
from pathlib import Path

from jaxdem.utils.particle_creation import create_ga_state
from jaxdem.utils.surface_properties import compute_surface_properties
from mu_utils import find_num_vertices_for_target_mu_eff_2d, get_closest_vertex_radius_for_mu_eff_2d

if __name__ == "__main__":
    mu_eff = float(sys.argv[1])

    out_dir = Path("/home/rg2248/project/friction/data/local-friction/uniform/sample")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_nv = 20
    overlap_fraction = 1e-5
    n_points = 5003
    n_orientations = 5009

    small_radius = 0.5
    large_radius = 0.7

    small_asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(
        mu_eff, small_radius, min_nv
    )
    large_asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(
        mu_eff, large_radius, min_nv
    )

    small_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(
        mu_eff,
        small_asperity_radius,
        small_radius,
        num_vertices_min=min_nv,
    )
    large_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(
        mu_eff,
        large_asperity_radius,
        large_radius,
        num_vertices_min=min_nv,
    )

    small = create_ga_state(
        N=1,
        nv=small_nv,
        dim=2,
        particle_radius=small_radius,
        asperity_radius=small_asperity_radius,
        particle_type="clump",
        core_type="solid",
        aspect_ratio=1.0,
        mesh_type="arclength",
        n_samples=10_000_000,
        seed=0,
    )

    large = create_ga_state(
        N=1,
        nv=large_nv,
        dim=2,
        particle_radius=large_radius,
        asperity_radius=large_asperity_radius,
        particle_type="clump",
        core_type="solid",
        aspect_ratio=1.0,
        mesh_type="arclength",
        n_samples=10_000_000,
        seed=0,
    )

    pairs = {
        "ss": (small, small, 2 * small_radius),
        "sl": (small, large, small_radius + large_radius),
        "ll": (large, large, 2 * large_radius),
    }

    distributions = {}

    for key, (central, tracer, diameter) in pairs.items():
        result = compute_surface_properties(
            central,
            tracer,
            target_overlap=overlap_fraction * diameter,
            n_points=n_points,
            n_orientations=n_orientations,
            separation_tolerance=1e-12,
        )
        distributions[key] = np.asarray(result["mu"]).ravel()

    output_path = out_dir / f"mu-{mu_eff}.npz"
    np.savez(output_path, **distributions)