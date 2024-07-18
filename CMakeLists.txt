cmake_minimum_required(VERSION 3.22)
project(test_pybind11 CXX)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

find_package(Python REQUIRED
                    COMPONENTS Interpreter Development )
find_package(pybind11 REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

pybind11_add_module(${PROJECT_NAME} SHARED
                    test_pybind11.cpp
                    CalculatingPi.cpp CalculatingPi.h
                    CalculatingPi_gpu.cu CalculatingPi_gpu.cuh
                    SieveOfEratosthenes.cpp SieveOfEratosthenes.h
                    utils.cpp utils.h
                    jobs_fifo_queue.h thread_pool.h
                    )

#set_target_properties(${PROJECT_NAME} PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_CUDA_COMPILER)
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Werror all-warnings --use_fast_math>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror -D HAS_CUDA>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
    )
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

