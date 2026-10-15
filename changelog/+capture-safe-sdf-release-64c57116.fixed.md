Fix CUDA graph captures that fail when the garbage collector frees a dropped model's SDF textures or a `CollisionPipeline`'s pinned host buffer during the capture.
