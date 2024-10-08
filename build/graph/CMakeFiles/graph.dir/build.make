# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuanxinyu/CUDA_project/CUDA_API

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuanxinyu/CUDA_project/CUDA_API/build

# Include any dependencies generated for this target.
include graph/CMakeFiles/graph.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include graph/CMakeFiles/graph.dir/compiler_depend.make

# Include the progress variables for this target.
include graph/CMakeFiles/graph.dir/progress.make

# Include the compile flags for this target's objects.
include graph/CMakeFiles/graph.dir/flags.make

graph/CMakeFiles/graph.dir/graph.cu.o: graph/CMakeFiles/graph.dir/flags.make
graph/CMakeFiles/graph.dir/graph.cu.o: ../graph/graph.cu
graph/CMakeFiles/graph.dir/graph.cu.o: graph/CMakeFiles/graph.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object graph/CMakeFiles/graph.dir/graph.cu.o"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/graph && /usr/local/cuda-12.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT graph/CMakeFiles/graph.dir/graph.cu.o -MF CMakeFiles/graph.dir/graph.cu.o.d -x cu -c /home/yuanxinyu/CUDA_project/CUDA_API/graph/graph.cu -o CMakeFiles/graph.dir/graph.cu.o

graph/CMakeFiles/graph.dir/graph.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/graph.dir/graph.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

graph/CMakeFiles/graph.dir/graph.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/graph.dir/graph.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target graph
graph_OBJECTS = \
"CMakeFiles/graph.dir/graph.cu.o"

# External object files for target graph
graph_EXTERNAL_OBJECTS =

graph/graph: graph/CMakeFiles/graph.dir/graph.cu.o
graph/graph: graph/CMakeFiles/graph.dir/build.make
graph/graph: /usr/local/cuda-12.2/lib64/libcudart.so
graph/graph: /usr/lib/x86_64-linux-gnu/libcuda.so
graph/graph: event/libEventPool.so
graph/graph: graph/CMakeFiles/graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable graph"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/graph && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
graph/CMakeFiles/graph.dir/build: graph/graph
.PHONY : graph/CMakeFiles/graph.dir/build

graph/CMakeFiles/graph.dir/clean:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/graph && $(CMAKE_COMMAND) -P CMakeFiles/graph.dir/cmake_clean.cmake
.PHONY : graph/CMakeFiles/graph.dir/clean

graph/CMakeFiles/graph.dir/depend:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanxinyu/CUDA_project/CUDA_API /home/yuanxinyu/CUDA_project/CUDA_API/graph /home/yuanxinyu/CUDA_project/CUDA_API/build /home/yuanxinyu/CUDA_project/CUDA_API/build/graph /home/yuanxinyu/CUDA_project/CUDA_API/build/graph/CMakeFiles/graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : graph/CMakeFiles/graph.dir/depend

