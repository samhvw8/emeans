# The emeans C quickstart, built through find_package(emeans).
#   cmake -S examples/c -B build/c -DCMAKE_PREFIX_PATH=$PWD/dist
#   cmake --build build/c && build/c/quickstart && build/c/quickstart_static
cmake_minimum_required(VERSION 3.20)
project(emeans_quickstart C)

find_package(emeans 0.1 REQUIRED)

add_executable(quickstart quickstart.c)
target_link_libraries(quickstart PRIVATE emeans::emeans m)

add_executable(quickstart_static quickstart.c)
target_link_libraries(quickstart_static PRIVATE emeans::emeans_static m)
