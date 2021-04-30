#ifndef MNIST_READER_CUH
#define MNIST_READER_CUH
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include "matrix.cuh"

using namespace std;

inline int reverseInt(int value);
void readLabels(string fileName, Matrix &labels, int len);
void readImages(string fileName, Matrix &images, int len);
unordered_map<string, Matrix> readData();

#endif