import jax
jax.config.update("jax_enable_x64", True)

from jaxdem.utils.h5 import load
from jaxdem.utils.contacts import compute_clump_pair_friction

import sys
import numpy as np
from pathlib import Path


def clump_outer_radii(state, n_clumps):
    clump_id = np.asarray(state.clump_id)
    pos_p = np.asarray(state.pos_p)
    rad = np.asarray(state.rad)

    return np.array([
        np.max(
            np.linalg.norm(pos_p[clump_id == i], axis=1)
            + rad[clump_id == i]
        )
        for i in range(n_clumps)
    ])


def mean_coordination_by_type(contact_mask, small):
    contact_mask = np.asarray(contact_mask, dtype=bool)
    small = np.asarray(small, dtype=bool)
    n_clumps = contact_mask.shape[0]

    z_all = contact_mask.sum(axis=1).astype(float)
    z_ss = np.zeros(n_clumps, dtype=float)
    z_sl = np.zeros(n_clumps, dtype=float)
    z_ll = np.zeros(n_clumps, dtype=float)

    i, j = np.triu_indices(n_clumps, 1)
    active = contact_mask[i, j]
    i, j = i[active], j[active]

    ss = small[i] & small[j]
    sl = small[i] ^ small[j]
    ll = ~small[i] & ~small[j]

    for mask, counts in ((ss, z_ss), (sl, z_sl), (ll, z_ll)):
        idx = np.concatenate([i[mask], j[mask]])
        counts += np.bincount(idx, minlength=n_clumps).astype(float)

    return {
        "all": float(z_all.mean()),
        "ss": float(z_ss.mean()),
        "sl": float(z_sl.mean()),
        "ll": float(z_ll.mean()),
    }


if __name__ == "__main__":
    mu_eff = float(sys.argv[1])

    data_root = Path("/home/rg2248/palmer_scratch/data/summer-26/jam-uniform/n-1000")
    out_dir = Path("/home/rg2248/project/friction/data/local-friction/uniform/jammed")
    out_dir.mkdir(parents=True, exist_ok=True)

    distributions = {"ss": [], "sl": [], "ll": [], "all": []}
    z_c = {"all": [], "ss": [], "sl": [], "ll": []}
    completed_runs = []

    for run in range(19):
        data_path = data_root / str(run) / f"mu-{mu_eff}"
        state_path = data_path / "state.h5"
        system_path = data_path / "system.h5"

        if not state_path.exists() or not system_path.exists():
            print(f"Skipping run {run}")
            continue

        print(f"Processing run {run}")

        state = load(state_path)
        system = load(
            system_path,
            state_shape=state.shape,
            warn_missing=False,
        )

        state, system, _, mu, contacts, _ = compute_clump_pair_friction(
            state,
            system,
            max_neighbors=512,
        )

        outer_radius = clump_outer_radii(state, mu.shape[0])
        small = outer_radius < (
            outer_radius.min() + outer_radius.max()
        ) / 2

        i, j = np.triu_indices(len(outer_radius), 1)
        active = np.asarray(contacts)[i, j]
        values = np.asarray(mu)[i, j]

        distributions["ss"].append(
            values[active & small[i] & small[j]]
        )
        distributions["sl"].append(
            values[active & (small[i] != small[j])]
        )
        distributions["ll"].append(
            values[active & ~small[i] & ~small[j]]
        )
        distributions["all"].append(values[active])

        for key, value in mean_coordination_by_type(contacts, small).items():
            z_c[key].append(value)

        completed_runs.append(run)

    if not completed_runs:
        print(f"No runs found for mu_eff={mu_eff}")
        sys.exit(0)

    output_path = out_dir / f"mu-{mu_eff}.npz"

    np.savez(
        output_path,
        **{
            key: np.concatenate(values)
            for key, values in distributions.items()
        },
        **{f"z_c_{key}": np.asarray(values) for key, values in z_c.items()},
        runs=np.asarray(completed_runs),
    )

    print(f"Saved {len(completed_runs)} runs to {output_path}")
