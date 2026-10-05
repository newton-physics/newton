# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure preparation only for one contact-dense world; not solver throughput."""

import argparse
import json
import statistics
import time

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.contact_torsion_device import DeviceTorsionPreparation
from newton.tests.test_feather_pgs_torsion_device import run_oracle, synthetic_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", type=int, nargs="+", default=[4, 64, 256, 1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for patches in (True, False):
        for count in args.contacts:
            reference = None
            for mode in ("host", "device", "graph"):
                solver, state, augmented, contacts = synthetic_fixture(
                    worlds=1, witnesses=count, patches=patches, device="cuda:0"
                )
                names = (
                    "constraint_count",
                    "slot_counter",
                    "row_type",
                    "row_parent",
                    "row_mu",
                    "row_beta",
                    "row_cfm",
                    "phi",
                    "target_velocity",
                    "row_restitution",
                    "_contact_torsion_group",
                )
                saved = [(getattr(solver, name), wp.clone(getattr(solver, name))) for name in names]
                saved += [(array, wp.clone(array)) for array in solver.J_by_size.values()]

                def restore(saved=saved):
                    for target, source in saved:
                        wp.copy(target, source)

                preparer = None if mode == "host" else DeviceTorsionPreparation(solver)
                if preparer is not None:
                    preparer.deferred_errors = mode == "graph"

                def prepare(preparer=preparer, solver=solver, state=state, augmented=augmented, contacts=contacts):
                    if preparer is None:
                        run_oracle(solver, state, augmented, contacts)
                    else:
                        preparer.prepare(state, augmented, contacts)

                restore()
                prepare()  # Compilation/warmup excluded.
                wp.synchronize_device("cuda:0")
                graph = None
                if mode == "graph":
                    restore()
                    wp.synchronize_device("cuda:0")
                    with wp.ScopedCapture(device="cuda:0") as capture:
                        prepare()
                    graph = capture.graph
                elapsed = []
                for _ in range(args.repeats):
                    restore()
                    wp.synchronize_device("cuda:0")
                    start = time.perf_counter()
                    if graph is None:
                        prepare()
                    else:
                        wp.capture_launch(graph)
                        preparer.validate()  # Mandatory error boundary included.
                    wp.synchronize_device("cuda:0")
                    elapsed.append((time.perf_counter() - start) * 1000.0)
                actual = {name: getattr(solver, name).numpy() for name in names}
                actual.update({f"J{key}": value.numpy() for key, value in solver.J_by_size.items()})
                if reference is None:
                    reference = actual
                else:
                    for key in reference:
                        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
                rows = int(np.count_nonzero(actual["row_type"] == 7))
                assert rows == 2, rows
                print(
                    json.dumps(
                        {
                            "worlds": 1,
                            "contacts": count,
                            "patches": patches,
                            "mode": mode,
                            "spin_rows": rows,
                            "median_ms": statistics.median(elapsed),
                            "repeats_ms": elapsed,
                            "parity": True,
                            "scope": "preparation_only_with_validation_excluding_restore",
                        }
                    ),
                    flush=True,
                )
    print("SINGLE_WORLD_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
