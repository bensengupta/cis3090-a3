#!/usr/bin/env bash

mkdir -p out/

python3 a3.py -s 9 2 test_files/b17_grayscale.png out/b17_grayscale_sharpened.png
python3 a3.py -s 9 2 test_files/b17_rgba.png out/b17_rgba_sharpened.png
python3 a3.py -s 9 2 test_files/citrus1.jpg out/citrus1_sharpened.jpg
python3 a3.py -s 9 2 test_files/usa_rgba.png out/usa_rgba_sharpened.png
python3 a3.py -s 9 2 test_files/vd-orig_rgba.png out/vd-orig_rgba_sharpened.png

python3 a3.py -n 9 4 test_files/b17_grayscale.png out/b17_grayscale_blurred.png
python3 a3.py -n 9 4 test_files/b17_rgba.png out/b17_rgba_blurred.png
python3 a3.py -n 9 4 test_files/citrus1.jpg out/citrus1_blurred.jpg
python3 a3.py -n 9 4 test_files/usa_rgba.png out/usa_rgba_blurred.png
python3 a3.py -n 9 4 test_files/vd-orig_rgba.png out/vd-orig_rgba_blurred.png
