Add `SolverMuJoCo.ObservableKind.QFRC_BIAS`, `QFRC_CONSTRAINT`, and
`QFRC_FRICTIONLOSS` solver observables. `qfrc_constraint - qfrc_frictionloss`
gives the constraint load on a DOF without its own friction loss, for example for
load-dependent gearbox friction models.
