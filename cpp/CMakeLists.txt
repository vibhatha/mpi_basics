cmake_minimum_required(VERSION 3.10)
project(MyMPIProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)

if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else(MPI_FOUND)
    message(SEND_ERROR "Application requires MPI")
    return()       
endif(MPI_FOUND)


add_library(utils STATIC src/util.cpp)

# Helper function to create a new target
function(create_target target_name source_files)
    add_executable(${target_name} ${source_files})
    target_link_libraries(${target_name} utils)
    target_link_libraries(${target_name} ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
      set_target_properties(${target_name} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
      set_target_properties(${target_name} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
endfunction()

# Example usage
create_target(hello_world "src/hello_world.cpp")
create_target(send_recv "src/send_recv.cpp")
create_target(round_robin "src/round_robin.cpp")
create_target(reduce "src/reduce.cpp")