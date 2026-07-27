#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/rust_dqn"
echo "Building DQN Rust extension..."
cargo build --release
echo "Done. Library at: target/release/libdqn_rust.so"
