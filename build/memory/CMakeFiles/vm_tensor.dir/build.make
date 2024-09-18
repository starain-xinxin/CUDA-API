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
include memory/CMakeFiles/vm_tensor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include memory/CMakeFiles/vm_tensor.dir/compiler_depend.make

# Include the progress variables for this target.
include memory/CMakeFiles/vm_tensor.dir/progress.make

# Include the compile flags for this target's objects.
include memory/CMakeFiles/vm_tensor.dir/flags.make

memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o: memory/CMakeFiles/vm_tensor.dir/flags.make
memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o: ../memory/vm_tensor.cu
memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o: memory/CMakeFiles/vm_tensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/memory && /usr/local/cuda-12.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o -MF CMakeFiles/vm_tensor.dir/vm_tensor.cu.o.d -x cu -c /home/yuanxinyu/CUDA_project/CUDA_API/memory/vm_tensor.cu -o CMakeFiles/vm_tensor.dir/vm_tensor.cu.o

memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/vm_tensor.dir/vm_tensor.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/vm_tensor.dir/vm_tensor.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target vm_tensor
vm_tensor_OBJECTS = \
"CMakeFiles/vm_tensor.dir/vm_tensor.cu.o"

# External object files for target vm_tensor
vm_tensor_EXTERNAL_OBJECTS =

memory/vm_tensor: memory/CMakeFiles/vm_tensor.dir/vm_tensor.cu.o
memory/vm_tensor: memory/CMakeFiles/vm_tensor.dir/build.make
memory/vm_tensor: /usr/local/cuda-12.2/lib64/libcudart.so
memory/vm_tensor: /usr/lib/x86_64-linux-gnu/libcuda.so
memory/vm_tensor: head_tool/libKernelForTest.so
memory/vm_tensor: memory/CMakeFiles/vm_tensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable vm_tensor"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/memory && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vm_tensor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
memory/CMakeFiles/vm_tensor.dir/build: memory/vm_tensor
.PHONY : memory/CMakeFiles/vm_tensor.dir/build

memory/CMakeFiles/vm_tensor.dir/clean:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/memory && $(CMAKE_COMMAND) -P CMakeFiles/vm_tensor.dir/cmake_clean.cmake
.PHONY : memory/CMakeFiles/vm_tensor.dir/clean

memory/CMakeFiles/vm_tensor.dir/depend:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanxinyu/CUDA_project/CUDA_API /home/yuanxinyu/CUDA_project/CUDA_API/memory /home/yuanxinyu/CUDA_project/CUDA_API/build /home/yuanxinyu/CUDA_project/CUDA_API/build/memory /home/yuanxinyu/CUDA_project/CUDA_API/build/memory/CMakeFiles/vm_tensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : memory/CMakeFiles/vm_tensor.dir/depend

