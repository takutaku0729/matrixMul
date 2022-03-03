// System includes
#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

#include <random>

void ConstantInitRand(float* data, int p, int pz, const dim3& size, int debug) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand100(1, 100);
   int infcount = 0;
   int count = 0;
   for (int y = 0; y < size.y / BLOCK_SIZE; y++) {
       for (int x = 0; x < size.x / BLOCK_SIZE; x++) {
           count++;
           if (rand100(rnd) < p) {
               for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                   for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                       data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = INF;
                   }
               }
               infcount++;
           }
           else {
#ifdef ZEROTILE
               if (rand100(rnd) < pz) {
                   for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                       for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 0;
                       }
                   }
               }
               else {
                   for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                       for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 1;
                       }
                   }
               }
#else
               for (int sy = 0; sy < BLOCK_SIZE; sy++) {
                   for (int sx = 0; sx < BLOCK_SIZE; sx++) {
                       if (rand100(rnd) < pz) {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 0;
                       }
                       else {
                           data[x * BLOCK_SIZE + y * size.x * BLOCK_SIZE + sx + sy * size.x] = 1;
                       }
                   }
               }
#endif
           }
       }
   }

    if(debug == 1){
        printf("inf percentage : %d / %d\n", infcount, count);
        int ex;
        for (int a = 0; a < size.y; a++) {
            for (int b = 0; b < size.x; b++) {
                ex = int(data[a * size.y + b]);
                printf("%d ", ex);
            }
            printf("\n");
        }
    }
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; i++) {
      data[i] = val;
    }
  }

void SetFileData(float* data, int size) {
    std::ifstream ifs("datas/" + FILENAME);

    if (!ifs) {
        std::cout << "Error: file not opened." << std::endl;
    }

    std::string tmp;
    for (int i = 0; i < 7; i++) {
        std::getline(ifs, tmp);
    }

    int count = 0;

    std::string buf;
    int x,y,value;
    for (int i = 0; i < size*size; i++) {
        ifs >> buf >> x >> y >> value;
        if (x < size && y < size) {
            data[x + y * size] = value;
            count++;
        }
    }
    count = size*size-count;
    printf("inf count %d / %d \n",count,size*size);
    ifs.close();
}

