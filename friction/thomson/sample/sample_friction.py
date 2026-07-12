import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
from pathlib import Path

from jaxdem.state import State
from jaxdem.utils.particle_creation import create_ga_state
from jaxdem.utils.surface_properties import compute_surface_properties
from mu_utils import (
    find_num_vertices_for_target_mu_eff_2d,
    get_closest_vertex_radius_for_mu_eff_2d,
    mean_edge_l_over_sigma_2d,
    mean_height_stats_2d,
)


def isolate_clump(state: State, clump_id: int = 0) -> State:
    """Extract a single clump from a multi-clump state and relabel it to id 0."""
    keep = jnp.asarray(state.clump_id) == clump_id
    if not bool(jnp.any(keep)):
        raise ValueError(f"clump_id {clump_id} not found in state")
    idx = jnp.where(keep)[0]
    isolated = jax.tree.map(lambda x: x[idx], state)
    isolated.clump_id = jnp.zeros(isolated.N, dtype=int)
    return isolated


if __name__ == "__main__":
    mu_eff = float(sys.argv[1])

    seed = 0
    particle_radius = 0.5
    min_nv = 90
    overlap_fraction = 1e-5
    n_points = 2003
    n_orientations = 2009

    out_dir = Path(f"/home/rg2248/project/friction/data/local-friction/thomson/sample/n-{min_nv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(
        mu_eff, particle_radius, min_nv
    )
    if not np.isfinite(asperity_radius):
        print(f"No achievable asperity radius for mu_eff={mu_eff}")
        sys.exit(1)

    nv, _, _ = find_num_vertices_for_target_mu_eff_2d(
        mu_eff,
        asperity_radius,
        particle_radius,
        num_vertices_min=min_nv,
    )
    if nv is None:
        print(f"No vertex count found for mu_eff={mu_eff}")
        sys.exit(1)

    ensemble = create_ga_state(
        N=1,
        nv=nv,
        dim=2,
        particle_radius=particle_radius,
        asperity_radius=asperity_radius,
        particle_type="clump",
        core_type="solid",
        n_samples=10_000_000,
        seed=seed,
        mesh_type="thomson",
        asperity_dispersity_type="uniform",
        asperity_dispersity_kwargs={"size_ratio": 2.0},
    )
    particle = isolate_clump(ensemble, clump_id=0)

    pos_p = np.asarray(particle.pos_p)
    rad = np.asarray(particle.rad)
    l_over_sigma, edge_l_over_sigma = mean_edge_l_over_sigma_2d(
        pos_p, rad, n_surface=nv
    )
    mean_height, mean_height_sq, height_std, height_sq_std = mean_height_stats_2d(
        pos_p, rad
    )
    outer_radius = float(np.max(np.linalg.norm(pos_p, axis=-1) + rad))
    diameter = 2.0 * outer_radius
    target_overlap = overlap_fraction * diameter

    result = compute_surface_properties(
        particle,
        particle,
        target_overlap=target_overlap,
        n_points=n_points,
        n_orientations=n_orientations,
        separation_tolerance=1e-12,
    )

    mu = np.asarray(result["mu"]).ravel()

    output_path = out_dir / f"mu-{mu_eff}.npz"
    np.savez(
        output_path,
        mu=mu,
        mu_eff=mu_eff,
        l_over_sigma=l_over_sigma,
        l_over_sigma_std=float(np.std(edge_l_over_sigma)),
        edge_l_over_sigma=edge_l_over_sigma,
        mean_height=mean_height,
        mean_height_sq=mean_height_sq,
        height_std=height_std,
        height_sq_std=height_sq_std,
        nv=nv,
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        seed=seed,
        n_points=n_points,
        n_orientations=n_orientations,
        target_overlap=target_overlap,
    )

    print(f"Saved {mu.size} surface friction samples to {output_path}")
