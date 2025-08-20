REM resize all PNGs to half size and convert to JPG
for f in *.png; do ffmpeg -y -i "$f" -vf "scale=iw/2:ih/2" -q:v 2 "${f%.png}.jpg"; done