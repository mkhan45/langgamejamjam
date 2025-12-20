#!/bin/sh
# cargo build --target=wasm32-unknown-unknown
# wasm-bindgen --target web --out-dir docs --no-typescript target/wasm32-unknown-unknown/debug/langame.wasm

source ../emsdk/emsdk_env.sh

CARGO_ARGS=""
RELEASE=false
if [ "$1" == "release" ]; then
    RELEASE=true
    CARGO_ARGS="--release"
fi

RUSTFLAGS="-Clink-args=-sEXPORTED_RUNTIME_METHODS=['cwrap','ccall','UTF8ToString','allocateUTF8']"
RUSTFLAGS="$RUSTFLAGS -Clink-args=-sMODULARIZE=1"
RUSTFLAGS="$RUSTFLAGS -Clink-args=-sEXPORT_ES6=1"
RUSTFLAGS="$RUSTFLAGS -Clink-args=-sALLOW_MEMORY_GROWTH=1"
export RUSTFLAGS

# cargo build --features profile --target=wasm32-unknown-emscripten $CARGO_ARGS
cargo build --target=wasm32-unknown-emscripten $CARGO_ARGS

if $RELEASE; then
    cp target/wasm32-unknown-emscripten/release/langame.wasm docs
    cp target/wasm32-unknown-emscripten/release/langame.js docs
else
    cp target/wasm32-unknown-emscripten/debug/langame.wasm docs
    cp target/wasm32-unknown-emscripten/debug/langame.js docs
fi
