#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. && cmake --build . 
echo "构建完成."

# cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 && cmake --build . 