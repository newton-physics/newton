@echo off
REM resize all PNGs to half size, crop to square, and convert to JPG
for %%f in (*.png) do (
    echo Processing %%f...
    REM First crop to square (keep as PNG), then resize to half and convert to JPG
    ffmpeg -y -i "%%f" -vf "crop=min(iw\,ih):min(iw\,ih)" -update 1 "%%~nf_temp.png"
    ffmpeg -y -i "%%~nf_temp.png" -vf "scale=iw/2:ih/2,format=yuv420p" -update 1 -q:v 2 "%%~nf.jpg"
    del "%%~nf_temp.png"
)