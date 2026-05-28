import sys
from pathlib import Path
from PIL import Image
import numpy as np
import warp as wp
import newton
import newton.examples

from newton.examples.contacts.example_hydro_shoe import Example

def main():
    parser = Example.create_parser()
    args = parser.parse_args([
        "--use-hydro-surface-wrench",
        "--viewer", "gl",
    ])

    viewer, args = newton.examples.init(parser)

    # Set camera target and position
    # Midsole center is at approximately (0.05, 0.15, 0.03)
    target = wp.vec3(0.05, 0.15, 0.03)
    pos = wp.vec3(0.45, -0.15, 0.25)
    
    # Position the camera and point it at the midsole center
    viewer.camera.pos = viewer.camera._as_vec3(pos)
    viewer.camera.look_at(viewer.camera._as_vec3(target))

    # Set visualization options
    viewer.show_hydro_contact_surface = True
    viewer.show_contacts = True
    viewer.show_visual = True

    example = Example(viewer, args)

    print("Running simulation to peak displacement...")
    peak_frame = 50
    for frame in range(peak_frame + 1):
        example.step()
        example.render()

    print("Capturing frame...")
    frame_data = viewer.get_frame(render_ui=True)
    frame_np = frame_data.numpy()

    output_path = "C:/Users/jkuzm/.gemini/antigravity-ide/brain/44f8a24d-1acc-4caf-8f9e-9b2642179a28/shoe_impact_render.png"
    img = Image.fromarray(frame_np)
    img.save(output_path)
    print(f"Captured frame successfully saved to {output_path}")

    viewer.close()

if __name__ == "__main__":
    main()
