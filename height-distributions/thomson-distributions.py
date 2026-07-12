import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sys
import numpy as np
from pathlib import Path

from jaxdem.state import State
from jaxdem.utils.particle_creation import create_ga_state
from mu_utils import (
    height_distribution_2d,
    mean_edge_l_over_sigma_2d,
    mean_height_stats_from_distribution,
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
    asperity_radius = float(sys.argv[1])

    rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**31 - 1))

    particle_radius = 0.5
    nv = 600
    n_angles = 100_000

    out_dir = Path(f"/home/rg2248/project/friction/data/height-distributions/thomson/n-{nv}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )
    particle = isolate_clump(ensemble, clump_id=0)

    pos_p = np.asarray(particle.pos_p)
    rad = np.asarray(particle.rad)
    angles, heights = height_distribution_2d(pos_p, rad, n_angles=n_angles)
    mean_height, mean_height_sq, height_std, height_sq_std = (
        mean_height_stats_from_distribution(heights)
    )
    l_over_sigma, edge_l_over_sigma = mean_edge_l_over_sigma_2d(
        pos_p, rad, n_surface=nv
    )

    output_path = out_dir / f"r-{asperity_radius:g}.npz"
    np.savez(
        output_path,
        angles=angles,
        heights=heights,
        mean_height=mean_height,
        mean_height_sq=mean_height_sq,
        height_std=height_std,
        height_sq_std=height_sq_std,
        l_over_sigma=l_over_sigma,
        l_over_sigma_std=float(np.std(edge_l_over_sigma)),
        edge_l_over_sigma=edge_l_over_sigma,
        nv=nv,
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        seed=seed,
        n_angles=n_angles,
    )

    print(f"Saved {heights.size} height samples to {output_path}")
