
import sys
import os
import numpy as np
import warp as wp
import argparse

# Add the repository root to sys.path to allow importing modules
repo_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import the module to test
from newton.examples.cosserat_codex.warp_cosserat_codex import Example, create_parser

def run_verification():
    print("Running verification of Warp implementation...")
    
    # Create dummy args
    parser = create_parser()
    args = parser.parse_args([])
    args.num_points = 20
    args.substeps = 1
    args.constraint_iterations = 1
    args.dll_path = None # Ensure it finds the DLL or fails gracefully if not found
    
    # Mock viewer
    class MockViewer:
        def __init__(self):
            self.show_particles = True
        def set_model(self, model):
            self.model = model
        def begin_frame(self, time):
            pass
        def end_frame(self):
            pass
        def log_state(self, state):
            pass
        def log_lines(self, name, start, end, color):
            pass
        def is_key_down(self, key):
            return False

    viewer = MockViewer()
    
    try:
        example = Example(viewer, args)
    except RuntimeError as e:
        print(f"Skipping verification: {e}")
        return

    # Enable Warp for all steps in numpy_rod
    print("Enabling Warp for all steps...")
    rod = example.numpy_rod
    for step in rod.warp_available:
        rod.set_warp_override(step, True)
    
    # Run a few steps
    num_steps = 10
    print(f"Running {num_steps} steps...")
    
    for i in range(num_steps):
        example.step()
        
        # Compare positions
        ref_pos = example.ref_rod.positions[:, 0:3]
        warp_pos = example.numpy_rod.positions[:, 0:3]
        
        # Account for offsets
        ref_pos_world = ref_pos + example.ref_offset
        warp_pos_world = warp_pos + example.numpy_offset
        
        # We compare relative to root to ignore offset
        ref_rel = ref_pos - ref_pos[0]
        warp_rel = warp_pos - warp_pos[0]
        
        diff = np.linalg.norm(ref_rel - warp_rel)
        max_diff = np.max(np.abs(ref_rel - warp_rel))
        
        print(f"Step {i}: Max diff between Reference and Warp = {max_diff:.6e}")
        
        if max_diff > 1e-4:
            print("FAILURE: Discrepancy too large!")
            # Print first few failing points
            err = np.linalg.norm(ref_rel - warp_rel, axis=1)
            indices = np.where(err > 1e-4)[0]
            print(f"Failing indices: {indices[:5]}")
            # return
            
    print("Verification completed.")

if __name__ == "__main__":
    run_verification()
