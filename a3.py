#!/usr/bin/env python3
# type: ignore

import sys
import numpy as np
import warp as wp

from PIL import Image


# Basic argument checking
if len(sys.argv) < 6:
    print(
        f"error: expected 6 arguments, received {len(sys.argv)} arguments",
        file=sys.stderr,
    )
    print(
        "usage: ./a3.py <algType> <kernSize> <param> <inFileName> <outFileName>",
        file=sys.stderr,
    )
    exit(1)

device = "cpu"

# Parse command line arguments
alg_type = sys.argv[1]
kernel_size = int(sys.argv[2])
param = float(sys.argv[3])
input_filename = sys.argv[4]
output_filename = sys.argv[5]

# Validate command line arguments
assert kernel_size >= 0, "Kernel size must positive"
assert kernel_size % 2 == 1, "Kernel size must be odd"


# Returns the value of the Gaussian matrix at the given position
# WARN: This matrix is NOT normalized
@wp.func
def gaussian_matrix_value(
    i: wp.int32, j: wp.int32, sigma: wp.float32, kernel_size: wp.int32
):
    x = wp.float32(i - kernel_size // 2)
    y = wp.float32(j - kernel_size // 2)

    return wp.exp(-(x * x + y * y) / (2.0 * sigma * sigma))


gaussian_matrix_norm = 0.0
for i in range(kernel_size):
    for j in range(kernel_size):
        gaussian_matrix_norm += gaussian_matrix_value(i, j, param, kernel_size)


@wp.kernel
def gaussian_blur_kernel(
    input: wp.array3d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    width, height = input.shape[0], input.shape[1]
    x, y, k = wp.tid()

    # Apply the kernel to the input pixel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Edge handling strategy: "extend"
            # https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_handling
            xi = wp.clamp(x + i - kernel_size // 2, 0, width - 1)
            yi = wp.clamp(y + j - kernel_size // 2, 0, height - 1)

            matrix_val = (
                gaussian_matrix_value(i, j, param, kernel_size) / gaussian_matrix_norm
            )

            output[x, y, k] += matrix_val * input[xi, yi, k]

    # Ensure pixel values are in the range [0, 255]
    output[x, y, k] = wp.clamp(output[x, y, k], 0.0, 255.0)


# Returns the value of the unsharp masking matrix at the given position,
# using box blur for blurring
@wp.func
def unsharp_masking_matrix_value(
    i: wp.int32, j: wp.int32, amount: wp.float32, kernel_size: wp.int32
):
    identity = 0.0
    if i == kernel_size // 2 and j == kernel_size // 2:
        identity = 1.0
    box_blur = 1.0 / wp.float32(kernel_size * kernel_size)

    return identity + amount * (identity - box_blur)


@wp.kernel
def unsharp_masking_kernel(
    input: wp.array3d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    width, height = input.shape[0], input.shape[1]
    x, y, k = wp.tid()

    # Apply the kernel to the input pixel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Edge handling strategy: "extend"
            # https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_handling
            xi = wp.clamp(x + i - kernel_size // 2, 0, width - 1)
            yi = wp.clamp(y + j - kernel_size // 2, 0, height - 1)

            matrix_val = unsharp_masking_matrix_value(i, j, param, kernel_size)

            output[x, y, k] += matrix_val * input[xi, yi, k]

    # Ensure pixel values are in the range [0, 255]
    output[x, y, k] = wp.clamp(output[x, y, k], 0.0, 255.0)


if alg_type == "-s":
    kernel = unsharp_masking_kernel
elif alg_type == "-n":
    assert param > 0, "sigma parameter must be positive"
    kernel = gaussian_blur_kernel
else:
    raise ValueError(f"Invalid algorithm type: {alg_type}")

input_image = Image.open(input_filename)
input = np.asarray(input_image, dtype=np.uint8)

if input_image.mode == "L":
    input = input[:, :, np.newaxis]

wp.set_module_options({"enable_backward": False})
wp.init()

input_wp = wp.array(input, dtype=wp.float32, device=device)
output_wp = wp.zeros(input_wp.shape, dtype=wp.float32, device=device)

wp.launch(
    kernel=kernel,
    dim=input_wp.shape,
    inputs=[input_wp, output_wp],
    device=device,
)

output = np.uint8(output_wp)

if input_image.mode == "L":
    output = output[:, :, 0]

output_image = Image.fromarray(output, mode=input_image.mode)
output_image.save(output_filename)
