cmake_minimum_required(VERSION 3.5)
project(tool_cloth_dynamics)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "set env var")

# Properly setting environment variables for Apple Clang
if (APPLE)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include" "/opt/homebrew/opt/libomp/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib" "/opt/homebrew/opt/libomp/lib")

    set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
    set(ENV{CXX} "/usr/local/opt/llvm/bin/clang-omp++")
    set(ENV{LDFLAGS} "-L/usr/local/opt/libomp/lib")
    set(ENV{CPPFLAGS} "-I/usr/local/opt/libomp/include")

    # Set OpenMP flags and libraries before find_package
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_C_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")

    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

    link_directories("/usr/local/Cellar/libomp/16.0.6/lib/")
endif()

if (MSVC)
    # Overwriting the flags with "-O3" would also drop MSVC's default /MD and
    # fall back to the static CRT, which does not link against slang-rhi.
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /MD")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_NAME tool_cloth_dynamics)
set(EXTERNAL_ROOT ${PROJECT_SOURCE_DIR}/external)
set(LIB dependencies/lib)

add_definitions(-DSOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}\")

add_definitions(-DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OpenMP_C "${CMAKE_C_COMPILER}")
  set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
  set(OpenMP_C_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "omp")

  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/16.0.6/include")
  set(OpenMP_CXX_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

  link_directories("/usr/local/Cellar/libomp/16.0.6/lib/")
endif()

#if (APPLE)
#    add_compile_options(-DUSING_MACBOOK)
#endif ()

if (USE_DEBUG)
    add_compile_options(-DUSE_DEBUG)
endif ()


message("cxx Flags: " ${CMAKE_CXX_FLAGS})

source_group("Shaders" FILES src/gpu/shaders)
source_group("Assets" FILES src/assets)

add_subdirectory(${EXTERNAL_ROOT})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)








###############
get_directory_property(EXTERNAL_HEADER
        DIRECTORY ${EXTERNAL_ROOT}
        DEFINITION EXTERNAL_HEADER)

include_directories(dependencies/include)
include_directories(src/assets)
include_directories(src/gpu)
include_directories(external/LBFGSpp/include)
include_directories(external/fcl/include)
include_directories(external/fcl/external/libccd/src)
include_directories(dependencies/lib)


link_directories(${LIB})

# OmegaEngine cpp
file(GLOB HEADERS "src/code/*.h" "src/code/engine/*.h" "src/code/simulation/*.h" "src/code/supports/*.h" "src/code/optimization/*.h" "src/code/slang_solver/*.h")
file(GLOB SRCFILES "src/code/main.cpp" "src/code/*.h" "src/code/*.c" "src/code/engine/*.cpp" "src/code/engine/*.h" "src/code/simulation/*.cpp" "src/code/simulation/*.h" "src/code/supports/*.cpp" "src/code/supports/*.h" "src/code/optimization/*.cpp" "src/code/optimization/*.h" "src/code/slang_solver/*.h")
source_group("Sources" FILES ${SRCFILES})

# Slang/Metal CG solver wrapper. Only on Apple — the .mm needs
# Objective-C++ + ARC + Metal/Foundation frameworks. Elsewhere the
# call sites #ifdef out and DiffCloth's Eigen LLT remains the only
# solver.
if (APPLE)
    set(SLANG_SOLVER_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/MetalCGSolver.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdSolver.mm")
    set_source_files_properties(${SLANG_SOLVER_SRC}
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    list(APPEND SRCFILES ${SLANG_SOLVER_SRC})
    # Pure-C++ AVBD backward shim (CHI-14 Brick C). Pulled into the
    # DiffCloth build so `Simulation::stepBackwardAvbd` (Brick D) can
    # call it from the optimisation path.
    list(APPEND SRCFILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdBackwardShim.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdIft.cpp")
endif()

# slang-rhi AVBD backend (Vulkan / D3D12 / CUDA, and Metal through slang-rhi).
# Replaces the Metal-only AvbdSolver.mm; build exactly one AvbdSolver. Needs a
# slang-rhi checkout that has been configured and built in <dir>/build.
#   cmake -DAVBD_USE_RHI=ON -DSLANG_RHI_DIR=/path/to/slang-rhi ...
option(AVBD_USE_RHI "Build the slang-rhi AVBD backend" OFF)
set(SLANG_RHI_DIR "" CACHE PATH "slang-rhi source dir (with a build/ subdir)")
if (AVBD_USE_RHI)
    if (NOT SLANG_RHI_DIR)
        message(FATAL_ERROR "AVBD_USE_RHI needs -DSLANG_RHI_DIR=<slang-rhi checkout>")
    endif()
    set(SLANG_RHI_BUILD "${SLANG_RHI_DIR}/build")
    set(SLANG_SRC "${SLANG_RHI_BUILD}/_deps/slang-src")
    if (APPLE)
        list(REMOVE_ITEM SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdSolver.mm")
    else()
        list(APPEND SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdBackwardShim.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdIft.cpp")
    endif()
    list(APPEND SRCFILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/RhiContext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver/AvbdSolverRhi.cpp")
endif()

add_executable(${EXECUTABLE_NAME} ${SRCFILES}  )

if (AVBD_USE_RHI)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE AVBD_USE_RHI=1)
    target_include_directories(${EXECUTABLE_NAME} PRIVATE
        "${SLANG_RHI_DIR}/include" "${SLANG_RHI_BUILD}/include" "${SLANG_SRC}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/code/slang_solver")
    target_link_directories(${EXECUTABLE_NAME} PRIVATE "${SLANG_RHI_BUILD}" "${SLANG_SRC}/lib")
    target_link_libraries(${EXECUTABLE_NAME} slang-rhi slang-rhi-resources slang-rhi-vma slang)
    # slang runtime DLLs next to the executable
    if (WIN32)
        file(GLOB SLANG_RHI_DLLS "${SLANG_RHI_BUILD}/*.dll")
        foreach(dll ${SLANG_RHI_DLLS})
            add_custom_command(TARGET ${EXECUTABLE_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>")
        endforeach()
    endif()
endif()

target_link_directories(${PROJECT_NAME} PRIVATE "/usr/local/Cellar/libomp/16.0.6/lib/")
target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${EXTERNAL_HEADER})

if (APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(${EXECUTABLE_NAME}
        ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()
