import jax
jax.config.update("jax_enable_x64", True)

from jaxdem.utils.h5 import load
from jaxdem.utils.contacts import compute_clump_pair_friction

import sys
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    mu_eff = float(sys.argv[1])

    data_root = Path("/home/rg2248/project/friction/data/jam-thomson/n-500")
    out_dir = Path("/home/rg2248/project/friction/data/local-friction/thomson/jammed")
    out_dir.mkdir(parents=True, exist_ok=True)

    distributions = {"ss": [], "sl": [], "ll": [], "all": []}
    completed_runs = []

    for run in range(20):
        data_path = data_root / f"mu-{mu_eff}" / str(run)
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

        clump_id = np.asarray(state.clump_id)
        pos_p = np.asarray(state.pos_p)
        rad = np.asarray(state.rad)

        outer_radius = np.array([
            np.max(
                np.linalg.norm(pos_p[clump_id == i], axis=1)
                + rad[clump_id == i]
            )
            for i in range(mu.shape[0])
        ])

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
        runs=np.asarray(completed_runs),
    )

    print(f"Saved {len(completed_runs)} runs to {output_path}")
