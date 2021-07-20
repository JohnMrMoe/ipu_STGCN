#ifndef ___MATRICES___J_
#define ___MATRICES___J_

#include <vector>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Everything uses float, we allow for no variation
using namespace std;

class Raw_Matrix {
public:
  float * data;
  size_t size  = 0;
  string title;
  Raw_Matrix(size_t n, string titled = "Untitled") : title(titled) {
    data = (float *) malloc (sizeof(float) * n);
    size = n;
  }
  ~Raw_Matrix() {
    //printf("Free alt [%ld, %ld, %ld]\n", data, &data, *data);
    //free(data);
  }
  float *ptr_to_raw() {
    return data;
  }
  void glorot_fill(size_t fan_in, size_t fan_out) {
    float limit = sqrt(6 / (fan_in + fan_out));
    float lim2 = limit*2;
    float r;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
      r = (float) rand();
      r /= 100;
      r -= (r-lim2);
      r -= limit;
      data[i] = r;
    }
  }
  void fill_mat(float val) {
    //fill(&data, &data + size, val);
    #pragma omp parallel for
    for (unsigned int i = 0; i < size; i++) {
      data[i]=val;
    }
  }
  size_t _size() {return size;}
  float mean() {
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < size; i++) {
      sum+=data[i];
    }
    return sum/size;
  }
  float standard_deviation(float mean) {
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < size; i++) {
      sum+=data[i]-mean;
    }
    return sum/size;
  }
  size_t nnz() {
    size_t _nnz;
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      _nnz += !data[i];
    }
    return _nnz;
  }
  float nnz_percent() {
    float _nnz =  nnz();
    float _size = (float) size;
    return _nnz/_size*100;
  }
  void z_score(float mean, float std) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      data[i] = (data[i] - mean) / std;
    }
  }
  void z_inverse(float *x, float mean, float std) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      data[i] = (x[i] * std) + mean;
    }
  }
};
class Matrix_2D : public Raw_Matrix {
public:
  float ** dereff;
  size_t outer, inner;
  Matrix_2D(size_t size, vector<size_t> shape, string titled="Untitled") : Raw_Matrix(size, titled) {
    outer = shape[0];
    inner = shape[1];

    dereff = (float **) malloc (sizeof (float*) * outer);
    size_t offset = 0;
    for (size_t i = 0; i < outer; i++, offset += inner) {
      dereff[i] = &data[offset];
    }
  }
  ~Matrix_2D() {
    // free(dereff);
  }
  float* operator[](size_t idx) {
    return dereff[idx];
  }

};
class Matrix_4D : public Raw_Matrix {
public:
  //float **** dereff;
  size_t W, Z, Y, X;
  size_t ZYX, YX;
  Matrix_4D(vector<size_t> shape, string titled="Untitled") : Raw_Matrix(shape[0]*shape[1]*shape[2]*shape[3]) {
    W=shape[0];
    Z=shape[1];
    Y=shape[2];
    X=shape[3];
    YX  = Y*X;
    ZYX = Z*YX;
  }
  float* operator[](size_t cube) {
    return &data[ZYX*cube];
  }
  size_t cube_nnz(size_t w, size_t z) {
    size_t sta = w*ZYX + z*YX;
    size_t sto = w*ZYX + z*YX + YX;
    size_t sum = 0;
    for (size_t i = sta; i < sto; i++) sum += !data[i];
    return sum;
  }
};
#endif
