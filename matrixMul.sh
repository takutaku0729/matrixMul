#!/bin/bash

nvcc matrixMul.cu setData.cu -o matrixMul && ./matrixMul $@