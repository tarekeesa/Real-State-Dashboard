#!/bin/bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install uv
pip install uv

# Use uv to install dependencies
uv install -r requirements.txt
