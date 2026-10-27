Correct full-surface mesh contact force directions when the narrow sign query
fails by propagating the expanded query's inside/outside classification. Keep
unresolved primitive pairs for rigid-soft DAT while excluding them from VBD
forces, including during contact-buffer overflow.
