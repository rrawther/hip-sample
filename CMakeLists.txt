cmake_minimum_required(VERSION 2.8.3)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(12_cmake)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

set(MY_SOURCE_FILES MatrixTranspose.cpp)
set(MY_TARGET_NAME MatrixTranspose)
set(MY_HIPCC_OPTIONS )
set(MY_HCC_OPTIONS )
set(MY_NVCC_OPTIONS )


if (BUILD_SHARED_LIBS)
    set(STATIC_OR_SHARED SHARED)
else()
    set(STATIC_OR_SHARED STATIC)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
set (CMAKE_LINKER "${HIP_PATH}/bin/hipcc_cmake_linker_helper")
#set (CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> ${HIP_PATH}/../hcc <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set (CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_LINKER> ${HIP_PATH}/../hcc <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_library(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS "${MY_HIPCC_OPTIONS}" HCC_OPTIONS "${MY_HCC_OPTIONS}" ${STATIC_OR_SHARED})
#add_library(${MY_TARGET_NAME} SHARED ${MY_SOURCE_FILES})
target_compile_definitions(${MY_TARGET_NAME} PRIVATE __HIP_PLATFORM_HCC__)

set (SOURCES main.cpp)

#add_custom_target(combined COMMAND ar -x $<TARGET_FILE:MatrixTranspose>
#                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#                DEPENDS MatrixTranspose
#                )
hip_add_executable (hip_cmake_test main.cpp)
target_compile_definitions(hip_cmake_test PRIVATE __HIP_PLATFORM_HCC__)
target_include_directories(hip_cmake_test
    PRIVATE
        $<BUILD_INTERFACE:${HIP_PATH}/include>
        $<BUILD_INTERFACE:${HIP_PATH}/../include>)
target_link_libraries (hip_cmake_test MatrixTranspose hip_hcc)

hip_add_executable (hiprtc_test saxpy.cpp)
set_source_files_properties(saxpy.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
target_compile_definitions(hiprtc_test PRIVATE __HIP_PLATFORM_HCC__)
target_include_directories(hiprtc_test
    PRIVATE
        $<BUILD_INTERFACE:${HIP_PATH}/include>
        $<BUILD_INTERFACE:${HIP_PATH}/../include>)
target_link_libraries (hiprtc_test hip_hcc hiprtc)
