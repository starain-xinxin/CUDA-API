#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build . 
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 && cmake --build . 
echo "构建完成."
