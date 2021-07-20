#ifndef MATH_UTIL_GUARD
#define MATH_UTIL_GUARD

#define HARDCODE_N 228

#include <vector>
#include <math.h>       /* exp */
#include <string>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#include "Matrices.cpp"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> laplacian;

using namespace std;

float sum_array(float * array, int len) {
  float sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum = array[i];
  }
  return sum;
}
bool non_bool(Matrix_2D &data) {
  float v;
  for (size_t i = 0; i < data.outer; i++) {
    for (size_t j = 0; j < data.inner; j++) {
      v = data[i][j];
      if (v !=0 || v !=1) return true;
    }
  }
  return false;
}
int weight_matrix(Matrix_2D &data, float sigma2=0.1, float epsilon=0.5, bool scaling=true) {
  if (!scaling) return 0;
  if (!non_bool(data)) return 0;

  float mask, w;

  #pragma omp parallel for
  for (size_t i = 0; i < data.outer; i++) {
    for (size_t j = 0; j < data.inner; j++) {
      mask = i==j ? 0 : 1;
      w  = data[i][j] / 10000;
      w *= w;
      w  = (w * -1) / sigma2;
      w = exp(w);
      data[i][j] = w * (w>=epsilon) * mask;
    }
  }

  return 1;
}
int scaled_laplacian(Matrix_2D &dst, Matrix_2D &src) {
  auto n = dst.outer;
  vector<float> d((size_t)src.outer);
  for (size_t i = 0; i < src.outer; i++)
    d[i] = sum_array(src[i], src.inner);

  laplacian L(src.outer, src.inner);

  #pragma omp parallel for
  for (size_t i = 0; i < src.outer; i++) {
    for (size_t j = 0; j < src.inner; j++) {
      L(i, j) = - src[i][j];
      if (d[i] <= 0 || d[j] <= 0) continue;
      L(i, j) /= sqrt(d[i] * d[j]);
    }
  }
  Eigen::EigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> eigen(L, false);

  Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> e_values = eigen.eigenvalues();

  float lambda_max = e_values(0, 0).real();
  for (auto i = 0; i < e_values.cols(); i++) {
    auto curr = e_values(0, i).real();
    lambda_max = (curr > lambda_max) ? curr : lambda_max;
  }

  float * data = dst.data;

  size_t at = 0;
  for (auto i = 0; i < src.outer; i++){
    for (auto j = 0; j < src.inner; j++, at++) {
      data[at] = 2 * L(i, j) / lambda_max;
    }
    data[i*n + i] -= 1;
  }

  return 1;
}
/*
 *
 * L: dimension: any, likely HARDCODE_N x HARDCODE_N
 *
 * Lk: inner dim = n*ks
 *
**/
int cheb_poly_approx(Matrix_2D &L,
                     int ks, size_t n,
                     Matrix_2D &Lk) {
  if (ks<=1) {
    printf("You shouldn't use this method with a ks value equal to or less than 1.\n");
    return -1;
  }

  size_t size = (size_t) (L.outer * L.inner);
  vector<size_t> shape = {(size_t) L.inner,
                                (size_t) L.outer};


  Matrix_2D _l0(size, shape);
  Matrix_2D _l1(size, shape);
  Matrix_2D _l2(size, shape);

  for   (auto i = 0; i < n; i++) {
    float *line = _l0[i];
    for (auto j = 0; j < n; j++) {
      line[j] = i==j;
    }
  }

  memcpy(_l1.data, L.data, L.size*sizeof(float));


  float *w_line, *l_line, *l0_line, *l1_line;
  for (auto list_slot = 2; list_slot < ks; list_slot++) {
    //L_list[list_slot] = Matrix_2D(size, shape);
    for   (auto i = 0; i < n; i++) {
      w_line  = _l2[i];
      l_line  = L[i];
      l0_line = _l0[i];
      l1_line = _l1[i];
      for (auto j = 0; j < n; j++) {
        w_line[j] = 2 * l_line[j] * l1_line[j] - l0_line[j];
      }
    }
    memcpy(_l0.data, _l1.data, size * sizeof(float));
    memcpy(_l1.data, _l2.data, size * sizeof(float));
  }

  // CONCATENATE
  //      Prep return vector:
  //      NOT CORRECT
  for (auto ks_n = 0; ks_n < ks; ks_n++) {
    for   (auto i = 0; i < L.inner; i++) {
      for (auto j = 0; j < L.outer; j++) {
        Lk[i][ks_n*n+j] = _l2[i][j];
      }
    }
  }





  return 1;
}



#endif
