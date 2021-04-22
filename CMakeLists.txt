cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(Convolution_Matrix LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE Convolution_Matrix_SRC include/*.h src/*.cpp)

add_library(COMMON STATIC "${Convolution_Matrix_SRC}")
target_include_directories(COMMON PUBLIC "${CMAKE_SOURCE_DIR}/include/")

set(main_project_dir "${CMAKE_SOURCE_DIR}/main/")
file(GLOB main_projects "${main_project_dir}/*.cu")
foreach(main_project_path ${main_projects})

	get_filename_component(main_project_name ${main_project_path} NAME)

	message("Main project path: ${main_project_path} and name: ${main_project_name}")
	
	add_executable("${main_project_name}" "${main_project_path}")
	target_link_libraries("${main_project_name}" PUBLIC COMMON)

endforeach()