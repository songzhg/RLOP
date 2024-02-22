# Lunar Lander

Lunar lander is a classic rocket trajectory optimization problem, which is part of the Box2D environments of gymnasium. For more detail, please refer to: https://gymnasium.farama.org/environments/box2d/lunar_lander/.

## Requirements

1. **Installation of libtorch:**
    Follow the instructions on the official PyTorch website: https://pytorch.org/cppdocs/installing.html. In this example, we select the GPU version of libtorch. Please modify the paths in the following part of CMakeList (examples/lunar_lander/CMakeList.txt):

    ```
    # libtorch gpu
    set(CUDA_TOOLKIT_ROOT_DIR "path/to/cuda")
    find_package(CUDA REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH "path/to/libtorch")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(lunar_lander PRIVATE "${TORCH_LIBRARIES}")
    set_property(TARGET lunar_lander PROPERTY CXX_STANDARD 17)
    if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:example-app>)
    endif (MSVC)
    ```

2. **Installation of Gymnasium:**
    ```
    pip install gymnasium[box2d]
    ```

3. **Installation of pybind11:**
    ```
    cd path/to/project
    mkdir third_party
    cd third_party
    git clone -b stable https://github.com/pybind/pybind11.git
    ```

## Run

1. **Compile**

    ```
    cd path/to/project
    mkdir build
    cd build
    cmake .. -DBUILD_LUNAR_LANDER=ON
    make
    ```
2. **Run**

    Run DQN.
    ```
    ./examples/lunar_lander/lunar_lander dqn
    ```

    Run PPO.
    ```
    ./examples/lunar_lander/lunar_lander ppo
    ```
    

