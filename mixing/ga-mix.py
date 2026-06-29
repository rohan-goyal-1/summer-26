import jax
jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
from pathlib import Path

from jaxdem.utils.particle_creation import create_ga_state
from jaxdem.utils.surface_properties import compute_surface_properties
from mu_utils import (
    find_num_vertices_for_target_mu_eff_2d,
    get_closest_vertex_radius_for_mu_eff_2d,
)

if __name__ == "__main__":
    mu_low = float(sys.argv[1])
    mu_high = float(sys.argv[2])
    assert mu_low <= mu_high, (
        f"expected mu_low <= mu_high, got {mu_low} > {mu_high}"
    )

    out_dir = Path("/home/rg2248/project/friction/data/local-friction/uniform/mix")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_nv = 20
    overlap_fraction = 1e-5
    n_points = 5003
    n_orientations = 5009
    seed = 0
    particle_radius = 0.5
    diameter = 2 * particle_radius

    def build_particle(mu_eff):
        asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(
            mu_eff, particle_radius, min_nv
        )
        if not np.isfinite(asperity_radius):
            raise ValueError(f"No achievable asperity radius for mu_eff={mu_eff}")

        nv, _, _ = find_num_vertices_for_target_mu_eff_2d(
            mu_eff,
            asperity_radius,
            particle_radius,
            num_vertices_min=min_nv,
        )
        if nv is None:
            raise ValueError(f"No vertex count found for mu_eff={mu_eff}")

        particle = create_ga_state(
            N=1,
            nv=nv,
            dim=2,
            particle_radius=particle_radius,
            asperity_radius=asperity_radius,
            particle_type="clump",
            core_type="solid",
            aspect_ratio=1.0,
            mesh_type="arclength",
            n_samples=10_000_000,
            seed=seed,
        )
        return particle, asperity_radius

    particle_low, asperity_radius_low = build_particle(mu_low)
    particle_high, asperity_radius_high = build_particle(mu_high)

    particles = {
        "low": particle_low,
        "high": particle_high,
    }

    pairs = {
        "mix_lh": ("low", "high"),
        "mix_hl": ("high", "low"),
    }

    distributions = {}

    for key, (central_key, tracer_key) in pairs.items():
        result = compute_surface_properties(
            particles[central_key],
            particles[tracer_key],
            target_overlap=overlap_fraction * diameter,
            n_points=n_points,
            n_orientations=n_orientations,
            separation_tolerance=1e-12,
        )
        distributions[key] = np.asarray(result["mu"]).ravel()

    output_path = out_dir / f"mu-{mu_low}-mu-{mu_high}.npz"
    np.savez(
        output_path,
        **distributions,
        mu_low=mu_low,
        mu_high=mu_high,
        asperity_radius_low=asperity_radius_low,
        asperity_radius_high=asperity_radius_high,
        particle_radius=particle_radius,
        seed=seed,
        n_points=n_points,
        n_orientations=n_orientations,
    )

    print(f"Saved mixed GA friction samples to {output_path}")
