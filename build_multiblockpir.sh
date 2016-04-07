#!/usr/bin/env bash

gcc -fPIC -O3 -Wall -Wextra -funroll-loops -march=native -c libmultiblockpir.c
gcc -shared -Wl,-soname,libmultiblockpir.so.1 -o libmultiblockpir.so.1 libmultiblockpir.o -lc -lrt -lm
