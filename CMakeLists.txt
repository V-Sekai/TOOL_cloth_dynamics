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

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
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

option(USE_SCHWARZ_PRECONDITIONER "Use Schwarz preconditioner for backward BiCGSTAB" OFF)
if (USE_SCHWARZ_PRECONDITIONER)
    add_compile_definitions(USE_SCHWARZ_PRECONDITIONER)
endif ()

option(USE_FCL "Use FCL for collision detection (mesh-based BVH and primitives)" OFF)
if (USE_FCL)
    add_compile_definitions(USE_FCL)
endif ()


message("cxx Flags: " ${CMAKE_CXX_FLAGS})

source_group("Shaders" FILES src/gpu/shaders)
source_group("Assets" FILES src/assets)

add_subdirectory(${EXTERNAL_ROOT})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# FCL (Flexible Collision Library) for collision detection when USE_FCL is ON
if (USE_FCL)
    set(EIGEN3_INCLUDE_DIR ${EXTERNAL_ROOT}/eigen CACHE PATH "Eigen3 include directory for FCL")
    set(BUILD_TESTING OFF CACHE BOOL "Disable FCL tests when building as subproject")
    set(FCL_BUILD_TESTS OFF CACHE BOOL "Disable FCL tests")
    # Build libccd first so FCL can link to target ccd
    add_subdirectory(thirdparty/libccd ${CMAKE_BINARY_DIR}/thirdparty/libccd)
    set(ccd_FOUND ON CACHE BOOL "libccd provided by thirdparty/libccd")
    add_subdirectory(thirdparty/fcl ${CMAKE_BINARY_DIR}/thirdparty/fcl)
endif()








###############
get_directory_property(EXTERNAL_HEADER
        DIRECTORY ${EXTERNAL_ROOT}
        DEFINITION EXTERNAL_HEADER)

include_directories(dependencies/include)
include_directories(src/assets)
include_directories(src/gpu)
include_directories(external/LBFGSpp/include)
if (USE_FCL)
    # FCL and libccd are provided by thirdparty/fcl build
else()
    include_directories(external/fcl/include)
    include_directories(external/fcl/external/libccd/src)
endif()
include_directories(external/preconditioner)
include_directories(dependencies/lib)


link_directories(${LIB})

# OmegaEngine cpp
file(GLOB HEADERS "src/code/*.h" "src/code/engine/*.h" "src/code/simulation/*.h" "src/code/supports/*.h" "src/code/optimization/*.h")
file(GLOB SRCFILES "src/code/main.cpp" "src/code/*.h" "src/code/*.c" "src/code/engine/*.cpp" "src/code/engine/*.h" "src/code/simulation/*.cpp" "src/code/simulation/*.h" "src/code/supports/*.cpp" "src/code/supports/*.h" "src/code/optimization/*.cpp" "src/code/optimization/*.h")
if (USE_FCL)
    list(APPEND SRCFILES src/code/simulation/CollisionFCL.cpp)
    list(APPEND HEADERS src/code/simulation/CollisionFCL.h)
endif()
# Schwarz preconditioner (Option B: copied sources, no submodule)
set(PRECONDITIONER_SOURCES
	external/preconditioner/SeOmp.cpp
	external/preconditioner/SeSchwarzPreconditioner.cpp
)
source_group("Sources" FILES ${SRCFILES})
source_group("Preconditioner" FILES ${PRECONDITIONER_SOURCES})
add_executable(${EXECUTABLE_NAME} ${SRCFILES} ${PRECONDITIONER_SOURCES})

target_link_directories(${PROJECT_NAME} PRIVATE "/usr/local/Cellar/libomp/16.0.6/lib/")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
if (USE_FCL)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE fcl)
endif()
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${EXTERNAL_HEADER})
