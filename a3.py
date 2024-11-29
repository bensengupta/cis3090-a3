#!/usr/bin/env python3
# type: ignore

import sys
import numpy as np
import warp as wp

from PIL import Image


def identity_kernel(kernel_size):
    """
    Returns a kernel that does not change the input image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1.0
    return kernel


def box_blur_kernel(kernel_size):
    """
    Returns a simple box blur kernel with the given size.
    """
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_kernel(kernel_size, sigma=4.0):
    """
    Returns a Gaussian kernel with the given size and standard deviation.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - kernel_size // 2
            y = j - kernel_size // 2
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def unsharp_masking_kernel(kernel_size, k=1.0):
    """
    Returns a kernel that sharpens the input image with unsharp masking
    using Box Blur.
    """
    return identity_kernel(kernel_size) + k * (
        identity_kernel(kernel_size) - box_blur_kernel(kernel_size)
    )


@wp.kernel
def convolve(
    input: wp.array3d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
    kernel: wp.array2d(dtype=wp.float32),
):
    kernel_size = kernel.shape[0]
    half_kernel_size = kernel.shape[0] // 2
    width, height = input.shape[0], input.shape[1]

    x, y, k = wp.tid()

    # Apply the kernel to the input pixel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Edge handling strategy: "extend"
            # https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_handling
            xi = wp.clamp(x + i - half_kernel_size, 0, width - 1)
            yi = wp.clamp(y + j - half_kernel_size, 0, height - 1)

            output[x, y, k] += kernel[i, j] * input[xi, yi, k]

    # Ensure pixel values are in the range [0, 255]
    output[x, y, k] = wp.clamp(output[x, y, k], 0.0, 255.0)


device = "cpu"

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

alg_type = sys.argv[1]
kernel_size = int(sys.argv[2])
param = float(sys.argv[3])
input_filename = sys.argv[4]
output_filename = sys.argv[5]

assert kernel_size >= 0, "Kernel size must positive"
assert kernel_size % 2 == 1, "Kernel size must be odd"

if alg_type == "-s":
    kernel = unsharp_masking_kernel(kernel_size, k=param)
elif alg_type == "-n":
    assert param > 0, "sigma parameter must be positive"
    kernel = gaussian_kernel(kernel_size, sigma=param)
else:
    raise ValueError(f"Invalid algorithm type: {alg_type}")

input_image = Image.open(input_filename)
input = np.asarray(input_image, dtype=np.uint8)

if input_image.mode == "L":
    input = input[:, :, np.newaxis]

wp.init()

input_wp = wp.array(input, dtype=wp.float32, device=device)
output_wp = wp.zeros(input_wp.shape, dtype=wp.float32, device=device)

kernel_wp = wp.array2d(kernel, dtype=wp.float32, device=device)
wp.launch(
    kernel=convolve,
    dim=input_wp.shape,
    inputs=[input_wp, output_wp, kernel_wp],
    device=device,
)

output = np.uint8(output_wp)

if input_image.mode == "L":
    output = output[:, :, 0]

output_image = Image.fromarray(output, mode=input_image.mode)
output_image.save(output_filename)
