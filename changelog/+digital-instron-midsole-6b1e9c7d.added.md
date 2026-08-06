Add the `digital_instron_midsole` example that turns the calibrated `projects/digital_instron_v2` column model into a fully dynamic viscoelastic midsole coupled into Newton rigid-body physics. It runs three scenarios that share one live Hyperfoam-Maxwell-Pasternak foundation:

  - `--mode instron`: a displacement-controlled digital Instron that squishes the midsole between a shoe-last crosshead and the ground plane and records the force-displacement hysteresis loop.
  - `--mode settle`: a free, massive midsole that rests in stable equilibrium on the foundation and resists a lateral load through Coulomb foam-shear friction.
  - `--mode stride`: a synthetic running stride that rolls a foot heel-to-toe over the foundation, producing a ground-reaction force profile and a migrating center of pressure.
