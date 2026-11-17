Fix gradient defects in the Featherstone differentiable path:

  - Deposit the dense Cholesky-solve adjoint on the lower triangle only and accumulate the armature-regularization gradient.
  - Stop `eval_rigid_tau` from overwriting its external-wrench input during the solve-frame shift.
  - Give each recorded step its own mass-matrix chain scratch so multi-step tape gradients stop aliasing (cost: roughly 2 x 4 B x (size of J + M + P + H + L) per world per recorded step, about 76 KB per world per step on a 13-joint articulation).
  - Keep the non-grad path's auxiliary arrays stable across gradient-recording steps and take the dense-solve adjoint path consistently.
