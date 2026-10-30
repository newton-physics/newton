Retain complete VT/TV/EE records in full-surface rigid-soft mesh collision,
including inactive soft features and all rigid triangle edges. VBD selects
forces at the current configuration, requiring a resolved inside/outside sign,
at least one active soft participant, and an active owner for shared TV features.
Unresolved pairs remain available to DAT but cannot apply VBD forces, even during
contact-buffer overflow. Use OGC's wireframe edge
region (Equation 14) for EE force selection. These additional records can require
more contact storage. If overflow is reported, stop the simulation and increase
`CollisionPipeline.soft_contact_max` to at least the printed requirement before
running again; DAT cannot protect against missing contacts.
