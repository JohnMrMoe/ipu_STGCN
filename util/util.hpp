#ifndef __UTIL
#define __UTIL



#include "Matrices.cpp"



float sum_array(float * array, int len);
bool non_bool(Matrix_2D &data);
int weight_matrix(Matrix_2D &data, float sigma2=0.1, float epsilon=0.5, bool scaling=true);
int scaled_laplacian(Matrix_2D &dst, Matrix_2D &src);
int cheb_poly_approx(Matrix_2D &L,
                     int ks, size_t n,
                     Matrix_2D &Lk);

#endif
