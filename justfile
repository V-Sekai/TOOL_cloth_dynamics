export URL := "https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_win64.exe.zip"
export FILENAME := "Godot_v4.3-stable_win64.exe.zip"
export EXTRACTED_FILE := "Godot_v4.3-stable_win64.exe"

build:
    mkdir build -p
    cd build && cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel

optimize demo randseed:
    @just build
    samply record ./build/DiffCloth -demo {{demo}} -mode optimize -seed {{randseed}}

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