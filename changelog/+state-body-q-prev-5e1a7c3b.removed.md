Remove the deprecated `State.body_q_prev` attribute. Solvers manage previous body transforms internally; applications that need pose history should clone `State.body_q` explicitly.
