# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# generate_icons.py

from PIL import Image, ImageDraw, ImageFont


def create_and_save_emoji_png(character: str, size: int, filename: str):
    """
    Renders a Unicode character onto a transparent PNG and saves it.
    """
    # Create a blank, transparent image
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Use a font that supports color emoji.
    # Windows: "Segoe UI Emoji" -> seguiemj.ttf
    # macOS: "Apple Color Emoji"
    # Linux: "Noto Color Emoji"
    font_path = "seguiemj.ttf"
    font_size = int(size * 0.8)  # Adjust font size relative to image size

    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        print(f"Warning: Font '{font_path}' not found. Using default font.")
        print("The icon may not render in color.")
        font = ImageFont.load_default()

    # Calculate position to center the character
    draw.textbbox((0, 0), character, font=font, spacing=0)
    x = size // 2
    y = size // 2 + 2  # +2 fudge factor

    # Draw the character onto the image
    draw.text((x, y), anchor="mm", text=character, font=font, embedded_color=True)

    # Save the image as a PNG file
    image.save(filename, "PNG")
    print(f"Successfully created {filename}")


if __name__ == "__main__":
    emoji_char = "🍏"
    sizes = [16, 32, 64]

    for s in sizes:
        output_filename = f"icon_{s}.png"
        create_and_save_emoji_png(emoji_char, s, output_filename)
