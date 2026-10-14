Accelerate experimental sparse Kamino DVI joint-friction and large batched
articulation solves by reusing sparse factorization structure and compact
Schur responses. Contact-free compact sweeps can stop early at exact fixed
points or bounded-row stationarity within the configured tolerance. Solver
tolerances, maximum iteration budgets, joint-friction laws, and contact laws
are unchanged. Schur-complement solves now ignore the bilateral solve interval
as documented instead of redundantly alternating bilateral solves; no
configuration migration is required.
