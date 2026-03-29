---
aliases:
  - world divergence
  - contact non-determinism
date: 2026-03-28
status: confirmed
commit: N/A
tags:
  - scaling
  - contact
  - mujoco-warp
  - upstream-issue
---

# World divergence: why identical worlds produce different physics

> [!bug] Upstream issue
> This is a MuJoCo Warp architecture limitation, not a CENIC bug. Fix requires per-world contact counters upstream.

Identically-initialized worlds diverge permanently after first contact event. This causes K variation across N and non-monotonic wall-time plots.

## The problem in plain terms

MuJoCo Warp detects contacts by having every world write into a shared contact buffer. When world 7 finds a contact between two objects, it needs a slot in the buffer to store that contact. It grabs the next available slot by incrementing a single shared counter.

The counter is shared across ALL N worlds. If world 7 and world 42 both find a contact at the same time, whoever grabs the counter first gets slot 0, the other gets slot 1. On a GPU, "at the same time" means "in the same clock cycle on different cores" -- and which core runs first depends on hardware scheduling that changes every run.

So the contact buffer ends up with the same contacts but in a **different order** depending on which world happened to run first. This matters because MuJoCo's constraint solver (PGS) processes constraints in buffer order. Processing contact A before contact B gives a slightly different result than B before A (the solver is iterative, not exact). The difference is tiny -- at the level of floating-point rounding -- but it's there.

## How a rounding-level difference becomes visible divergence

1. **Step 45** (first contact): worlds get contacts in different buffer order. Physics differs by ~1e-25 (one bit of floating point).
2. **Step 46**: that tiny force difference means objects are in very slightly different positions. During contact, forces are stiff -- small position changes produce large force changes. Difference jumps to ~1e-10.
3. **Step 47**: difference is now ~1e-6. CENIC's step doubling sees different error for each world. Some worlds reject the step, others don't. K diverges.
4. **Step 66+**: worlds are in completely different physical states. Permanently diverged.

The amplification is not a bug -- it's chaotic dynamics. Contact is inherently sensitive to initial conditions. The "initial condition" difference is just the buffer ordering.

## Why this affects K and wall time plots

K (iterations per step_dt) depends on which worlds reject steps. With more worlds, there's a higher chance at least one world is in a state that triggers a rejection. The boundary loop waits for ALL worlds, so one straggler world at K=11 makes the entire step_dt take 11 iterations even though the other 2047 worlds finished in 3.

This is why:
- K_mean increases slightly with N (2.6 at N=1, 3.2 at N=2048)
- K_max spikes happen at all N values (K_max=9-14 everywhere)
- Wall-time plots are non-monotonic (K varies randomly between N values)
- Per-iteration plots ARE monotonic (K is factored out)

## The shared counter (for reference)

```python
# collision_core.py:210 -- all N worlds share one counter
cid = wp.atomic_add(nacon_out, 0, 1)  # grabs next slot

# collision_driver.py:342 -- same pattern
pairid = wp.atomic_add(ncollision_out, 0, 1)
```

A fix would use per-world counters: `nacon_out[world_id]` instead of `nacon_out[0]`. This is an upstream MuJoCo Warp change.

## Implications

- World divergence is expected and unavoidable with the current MuJoCo Warp architecture
- Per-iteration cost is the correct scaling metric (factors out K variation)
- Use `--num-worlds 1` for visualization; use `--headless` for data collection at N > 1
- K_max=11-14 happens at all N values -- it's not N-dependent

Related: [[2026-03-28-sync-based-measured|current architecture]], [[2026-03-12-benchmark-infrastructure|benchmark infrastructure]], [[2026-03-13-drake-error-controller|Drake controller]].
