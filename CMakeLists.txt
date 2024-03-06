cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project ("rlop")

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11)

option(BUILD_VRP "Build VRP" OFF)
option(BUILD_SNAKE "Build Snake" OFF)
option(BUILD_LUNAR_LANDER "Build lunar lander" OFF)
option(BUILD_CONINUOUS_LUNAR_LANDER "Build continous lunar lander" OFF)
option(BUILD_CONNECT4 "Build connect4" OFF)
option(BUILD_MULTI_ARMED_BANDIT "Build multi-amred bandit" OFF)

if(BUILD_VRP)
    add_subdirectory(examples/vrp)
endif()

if(BUILD_SNAKE)
    add_subdirectory(examples/snake)
endif()

if(BUILD_LUNAR_LANDER)
    add_subdirectory(examples/lunar_lander)
endif()

if(BUILD_CONINUOUS_LUNAR_LANDER)
    add_subdirectory(examples/continuous_lunar_lander)
endif()

if(BUILD_CONNECT4)
    add_subdirectory(examples/connect4)
endif()

if(BUILD_MULTI_ARMED_BANDIT)
    add_subdirectory(examples/multi_armed_bandit)
endif()

# add_subdirectory(test/dqn/lunar_lander)
# add_subdirectory(test/ppo/lunar_lander)
# add_subdirectory(test/sac/continuous_lunar_lander)
# add_subdirectory(test/ppo/continuous_lunar_lander)