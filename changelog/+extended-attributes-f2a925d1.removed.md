Remove the built-in solver-produced extended attributes of `State` and `Contacts`, deprecated in Newton 1.7, together with the APIs that requested them. Solvers now write these arrays only to `SolverBase.Observables`.

**Migration:**

  - Replace `State.body_qdd`, `State.body_parent_f`, and `State.mujoco.qfrc_actuator` with `observables.body_qdd`, `observables.body_parent_f`, and `observables.qfrc_actuator`, requested with `SolverBase.ObservableKind.BODY_QDD`, `BODY_PARENT_F`, and `SolverMuJoCo.ObservableKind.QFRC_ACTUATOR`.
  - Replace `Contacts.force` with `observables.contact_f`, requested with `SolverBase.ObservableKind.CONTACT_F` and `solver.observables(kinds=..., contacts=contacts)`.
  - Replace `Model.request_state_attributes()`, `ModelBuilder.request_state_attributes()`, `Model.request_contact_attributes()`, and `ModelBuilder.request_contact_attributes()` with `solver.observables(kinds=...)`, and pass the container to `solver.step(..., observables=observables)`.
  - Remove the `requested_attributes` argument of `Contacts` and calls to `Model.get_requested_contact_attributes()`, and stop using `State.EXTENDED_ATTRIBUTES`, `State.EXTENDED_ATTRIBUTE_TEMPLATES`, `State.ExtendedAttributeTemplate`, `State.validate_extended_attributes()`, `Contacts.EXTENDED_ATTRIBUTES`, and `Contacts.validate_extended_attributes()`.
  - Replace force export through `update_contacts()` with contact observables passed to `step()`. `SolverMuJoCo.update_contacts()` and `SolverKamino.update_contacts()` keep exporting contact geometry; `SolverXPBD.update_contacts()` is removed, so calling it raises `NotImplementedError`.
