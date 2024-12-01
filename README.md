# CIS3090 Assignment 3

Benjamin Sengupta and Myron Ladyjenko

Introduction to GPGPU with Nvidia Warp: Image Processing.

## Pre-requisites

[uv](https://github.com/astral-sh/uv) OR Python 3.12+

## Setup

(Option 1) Within the cis3090 docker container:

```bash
# will install Python venv, create a new venv
# will use the newly created venv, and install required packages
./install.sh
```

(Option 2) With python:

```bash
pip install numpy warp-lang pillow
```

(Option 3) With uv:

```bash
uv sync
./.venv/bin/activate

# when finished with running program, run `deactivate`
```

## Usage

```
# usage: ./a3.py <algType> <kernSize> <param> <inFileName> <outFileName>
$ ./a3.py -s 5 4.0 in.png out.png
Warp 1.4.1 initialized:
   CUDA not enabled in this build
   Devices:
     "cpu"      : "arm"
   Kernel cache:
     /Users/bsengupta/Library/Caches/warp/1.4.1
Module __main__ 6d08326 load on device 'cpu' took 1.74 ms  (cached)
```
