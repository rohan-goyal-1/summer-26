import jax
jax.config.update("jax_enable_x64", True)

from jaxdem import fire
from jaxdem.utils.particle_creation import build_ga_system
from jaxdem.utils.jamming import bisection_jam
from jaxdem.utils.h5 import save

from mu_utils import find_num_vertices_for_target_mu_eff_2d, get_closest_vertex_radius_for_mu_eff_2d

from numpy.random import default_rng
from os import makedirs
from os.path import join
import sys

if __name__ == "__main__":
    mu_eff = float(sys.argv[1])
    run_id = int(sys.argv[2])

    rng = default_rng()
    n_clumps = 1000
    min_nv = 20
    rat_small = 0.5

    n_small = int(n_clumps * rat_small)
    n_large = n_clumps - n_small

    particle_radii = [0.5] * n_small + [0.7] * n_large

    small_asperity_rad = get_closest_vertex_radius_for_mu_eff_2d(mu_eff, 0.5, min_nv)
    large_asperity_rad = get_closest_vertex_radius_for_mu_eff_2d(mu_eff, 0.7, min_nv)
    asperity_radii = [small_asperity_rad] * n_small + [large_asperity_rad] * n_large

    small_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(mu_eff, small_asperity_rad, 0.5, num_vertices_min=min_nv)
    large_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(mu_eff, large_asperity_rad, 0.7, num_vertices_min=min_nv)
    vertex_counts = [small_nv] * n_small + [large_nv] * n_large

    state, system = build_ga_system(
        particle_radii=particle_radii,
        vertex_counts=vertex_counts,
        asperity_radius=asperity_radii,
        phi=0.4,
        dim=2,
        particle_type="clump",
        aspect_ratio=1.0,
        core_type="solid",
        domain_type="periodic",
        randomize_orientation=True,
        n_property_samples=10_000_000,
        mesh_type="arclength",
        compression_step=1e-2,
        max_n_min_steps_per_outer=50_000,
        dt=1e-2,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        minimizer=fire,
        minimizer_kw={"dt": 1e-2},
        collider_type="cell_list",
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    result = bisection_jam(state, system)
    state, system = result.jammed_state, result.jammed_system
    print(
        f"Jammed: phi = {float(result.packing_fraction):.6f}, "
        f"residual PE = {float(result.potential_energy):.3e}"
    )

    data_root = f'/home/rg2248/palmer_scratch/data/summer-26/jam-uniform/n-{n_clumps}/{run_id}/mu-{mu_eff}'
    makedirs(data_root, exist_ok=True)
    save(state, join(data_root, 'state.h5'))
    save(system, join(data_root, 'system.h5'))
