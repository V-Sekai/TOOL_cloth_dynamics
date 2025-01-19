export URL := "https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_win64.exe.zip"
export FILENAME := "Godot_v4.3-stable_win64.exe.zip"
export EXTRACTED_FILE := "Godot_v4.3-stable_win64.exe"
export LDFLAGS :="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS :="-I/opt/homebrew/opt/libomp/include"

build:
    #!/bin/bash
    if [[ `uname` == "Darwin" ]]; then
        brew install libomp
    fi
    mkdir build -p
    cd build && cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
    cmake --build . --parallel

optimize demo randseed:
    @just build
    ./build/DiffCloth -demo {{demo}} -mode optimize -seed {{randseed}}

optimize-all randseed:
    @just build
    #!/usr/bin/env -S parallel --shebang-wrap /bin/bash --ungroup --jobs $(nproc)
    echo "Starting optimization for T-shirt"; ./build/DiffCloth -demo tshirt -mode optimize -seed {{randseed}}; echo "T-shirt optimization done"
    echo "Starting optimization for Sphere"; ./build/DiffCloth -demo sphere -mode optimize -seed {{randseed}}; echo "Sphere optimization done"
    echo "Starting optimization for Hat"; ./build/DiffCloth -demo hat -mode optimize -seed {{randseed}}; echo "Hat optimization done"
    echo "Starting optimization for Sock"; ./build/DiffCloth -demo sock -mode optimize -seed {{randseed}}; echo "Sock optimization done"
    echo "Starting optimization for Dress"; ./build/DiffCloth -demo dress -mode optimize -seed {{randseed}}; echo "Dress optimization done"

clean:
    rm -rf build

fetch-and-extract:
    curl -o $FILENAME -L $URL
    unzip $FILENAME $EXTRACTED_FILE
    mv $EXTRACTED_FILE ./godot.exe
    rm $FILENAME
    @just godot

godot:
    ./godot.exe --headless --script hello_editor_script.gd --quit
    ./godot.exe --script obj_sequence_gltf.gd --quit