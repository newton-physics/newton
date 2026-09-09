Add regularized Coulomb friction for revolute, prismatic, and D6 joints in VBD, including joints in mimic relationships.

Use a stable friction linearization near stopping and joint-coordinate gradients for rotating frames and multi-axis joints. VBD's mimic solve uses the assembled body Hessians and retains constraint reactions so follower friction contributes to force balance. Friction is smoothed near zero speed, allowing slow creep rather than exact static sticking.
