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
include event/CMakeFiles/EventForSyn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include event/CMakeFiles/EventForSyn.dir/compiler_depend.make

# Include the progress variables for this target.
include event/CMakeFiles/EventForSyn.dir/progress.make

# Include the compile flags for this target's objects.
include event/CMakeFiles/EventForSyn.dir/flags.make

event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o: event/CMakeFiles/EventForSyn.dir/flags.make
event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o: ../event/EventForSyn.cu
event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o: event/CMakeFiles/EventForSyn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/event && /usr/local/cuda-12.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o -MF CMakeFiles/EventForSyn.dir/EventForSyn.cu.o.d -x cu -c /home/yuanxinyu/CUDA_project/CUDA_API/event/EventForSyn.cu -o CMakeFiles/EventForSyn.dir/EventForSyn.cu.o

event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/EventForSyn.dir/EventForSyn.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/EventForSyn.dir/EventForSyn.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target EventForSyn
EventForSyn_OBJECTS = \
"CMakeFiles/EventForSyn.dir/EventForSyn.cu.o"

# External object files for target EventForSyn
EventForSyn_EXTERNAL_OBJECTS =

event/EventForSyn: event/CMakeFiles/EventForSyn.dir/EventForSyn.cu.o
event/EventForSyn: event/CMakeFiles/EventForSyn.dir/build.make
event/EventForSyn: /usr/local/cuda-12.2/lib64/libcudart.so
event/EventForSyn: /usr/lib/x86_64-linux-gnu/libcuda.so
event/EventForSyn: event/CMakeFiles/EventForSyn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuanxinyu/CUDA_project/CUDA_API/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable EventForSyn"
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/event && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EventForSyn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
event/CMakeFiles/EventForSyn.dir/build: event/EventForSyn
.PHONY : event/CMakeFiles/EventForSyn.dir/build

event/CMakeFiles/EventForSyn.dir/clean:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build/event && $(CMAKE_COMMAND) -P CMakeFiles/EventForSyn.dir/cmake_clean.cmake
.PHONY : event/CMakeFiles/EventForSyn.dir/clean

event/CMakeFiles/EventForSyn.dir/depend:
	cd /home/yuanxinyu/CUDA_project/CUDA_API/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanxinyu/CUDA_project/CUDA_API /home/yuanxinyu/CUDA_project/CUDA_API/event /home/yuanxinyu/CUDA_project/CUDA_API/build /home/yuanxinyu/CUDA_project/CUDA_API/build/event /home/yuanxinyu/CUDA_project/CUDA_API/build/event/CMakeFiles/EventForSyn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : event/CMakeFiles/EventForSyn.dir/depend
