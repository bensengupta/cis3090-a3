#!/usr/bin/env bash

mkdir -p out/
rm out/*

python3 a3.py -n 3 1 test_files/citrus-cropped.png out/citrus-cropped_blur_3_1.png
python3 a3.py -n 3 4 test_files/citrus-cropped.png out/citrus-cropped_blur_3_4.png
python3 a3.py -n 13 1 test_files/citrus-cropped.png out/citrus-cropped_blur_13_1.png
python3 a3.py -n 13 4 test_files/citrus-cropped.png out/citrus-cropped_blur_13_4.png

python3 a3.py -s 3 1 test_files/citrus-cropped.png out/citrus-cropped_sharp_3_1.png
python3 a3.py -s 3 4 test_files/citrus-cropped.png out/citrus-cropped_sharp_3_4.png
python3 a3.py -s 13 1 test_files/citrus-cropped.png out/citrus-cropped_sharp_13_1.png
python3 a3.py -s 13 4 test_files/citrus-cropped.png out/citrus-cropped_sharp_13_4.png

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
