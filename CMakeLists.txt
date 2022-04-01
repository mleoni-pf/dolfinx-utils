cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME DOLFINX-UTILS)
project(${PROJECT_NAME})

# Get DOLFINX configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
  find_package(Basix REQUIRED)
endif()

add_subdirectory(src)
