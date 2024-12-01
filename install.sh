#!/bin/bash

sudo apt update
sudo apt install -y python3.11-venv
python3 -m venv venv
source venv/bin/activate
pip install numpy pillow warp-lang
