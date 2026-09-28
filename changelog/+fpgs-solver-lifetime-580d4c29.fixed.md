Remove an unused FeatherPGS debug-state ownership cycle so stepped solvers release their CUDA streams before cyclic garbage collection can finalize those streams first.
