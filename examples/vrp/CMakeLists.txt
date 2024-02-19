cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project ("vrp")

include_directories(${CMAKE_SOURCE_DIR})

add_executable (vrp "main.cc")