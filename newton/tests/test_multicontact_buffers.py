# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples


class TestMulticontactBuffers(unittest.TestCase):
    """Regression tests for stack-buffer usage in ``create_build_manifold``.

    The GJK/MPR manifold builder allocates its contact-polygon scratch space as
    Warp stack arrays (``wp.zeros`` -> ``fixedarray``) and passes them to shared
    ``@wp.func`` helpers. Two CUDA-only idioms previously made this code depend
    on ``fixedarray`` implicitly converting to ``array_t``:

    1. aliasing a device ``array_t`` view over a stack buffer via raw pointer
       arithmetic (``wp.array(ptr=buffer.ptr + offset, ...)``), and
    2. annotating the helper parameters as ``wp.array[wp.vec2]``.

    On backends without that implicit conversion (e.g. the Metal backend, where
    a ``device`` pointer may not address thread-local storage and ``fixedarray``
    does not derive from ``array_t``) these failed to compile. The helpers now
    take ``Any`` and the alias is replaced by an independent stack buffer, so
    the manifold kernels build on any Warp device.

    These tests are backend-agnostic: they exercise the manifold path on the
    default device and assert the collision kernels compile and produce
    contacts, which is what breaks when the fixedarray handling regresses.
    """

    def _build_quadruped(self):
        builder = newton.ModelBuilder()
        single = newton.ModelBuilder()
        single.add_urdf(
            open(newton.examples.get_asset("quadruped.urdf")).read(),
            joint_damping_mode="drive",
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3)),
        )
        single.add_ground_plane()
        builder.add_world(single)
        return builder.finalize(skip_validation_inertia=True)

    def test_manifold_kernels_compile_and_collide(self):
        """Cylinder/capsule-vs-plane collision routes through create_build_manifold."""
        model = self._build_quadruped()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()
        for _ in range(30):
            pipeline.collide(state, contacts)
        wp.synchronize_device(model.device)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "expected manifold contacts for a settled quadruped")

    def test_manifold_rollout_stays_finite(self):
        """A short Featherstone rollout through the manifold path stays finite.

        Guards the buffer rewrite against silent corruption of contact-point
        extraction (e.g. reading uninitialized slots of the resized buffers).
        """

        def rollout(device):
            wp.set_device(device)
            model = self._build_quadruped()
            solver = newton.solvers.SolverFeatherstone(model)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            s0, s1 = model.state(), model.state()
            for _ in range(10):
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, 0.002)
                s0, s1 = s1, s0
            wp.synchronize_device(device)
            return s0.body_q.numpy().astype(np.float64)

        cpu_state = rollout("cpu")
        self.assertFalse(np.isnan(cpu_state).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
