Speed up CUDA convex narrow-phase collision in large replicated scenes, without changing the produced contacts, by restricting the one-warp split GJK/MPR launch to models that carry convex support acceleration data; models using the exhaustive support scan keep the full block size, which also restores the thread count the rest of the narrow phase uses.

No migration is needed.
