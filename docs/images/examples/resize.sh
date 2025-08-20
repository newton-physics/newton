# resize all pngs in a directory by 2x in dimension + convert to jpg
for f in *.png; do ffmpeg -y -i "$f" -vf "scale=iw/2:ih/2" -q:v 3 "${f%.png}.jpg"; done