Start the scene editor paused, defer MuJoCo Warp compilation and solver CUDA graph capture until playback, and reuse captures across live pose and material edits. Increase ground friction and add rolling resistance to authored spheres. Preserve workspace selection when deferred physics becomes ready and expose readiness through the optional MCP bridge.

Reuse unchanged mesh boundary topology and matching GL geometry during model replacement to speed up primitive placement in mesh-heavy scenes.
