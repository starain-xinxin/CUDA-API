#!/bin/bash
cd build/test
echo "----------- Now run the all executable -----------"
for file in ./*; do
  if [[ -x "$file" && -f "$file" ]]; then
    echo " "
    echo " running $file ... "
    ./"$file"
    echo " "
  fi
done
