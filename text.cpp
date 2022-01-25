#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fstream>

int main(void){
  std::ofstream writing_file;
  std::string filename;
  filename = "test.csv";
  writing_file.open(filename, std::ios::out);
  writing_file << "test\n";
  writing_file << "test2\n";
  writing_file << "test3";
  exit(0);
}