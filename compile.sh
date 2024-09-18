#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. && cmake --build .
echo "构建完成."
