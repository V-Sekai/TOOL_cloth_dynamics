# Justfile for building and running quadwild on a Blender Monkey

build:
    mkdir -p build
    cd build && cmake .. && cmake --build . --parallel

optimize demo randseed:
    @just build
    ./build/DiffCloth -demo {{demo}} -mode optimize -seed {{randseed}}

visualize demo expName:
    @just optimize {{demo}} {{expName}}
    ./build/DiffCloth -demo {{demo}} -mode visualize -exp {{expName}}

clean:
    rm -rf build
