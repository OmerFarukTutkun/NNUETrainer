#ifndef _BASIC_MATRIX_OPERATIONS
#define _BASIC_MATRIX_OPERATIONS
#include "matrix.h"

void ElementwiseMultipyMatrix(Matrix* a, Matrix* b, Matrix* c);
void MultipyMatrix(Matrix* a, Matrix* b, Matrix* c);
void MultipyMatrix_abT(Matrix* a, Matrix* b, Matrix* c);
void addMatrix(Matrix* a, Matrix* b, Matrix* c);
void scale_and_addMatrix(Matrix*a, Matrix* b, Matrix* c, float x);
void subMatrix(Matrix* a, Matrix* b, Matrix* c);
float traceMatrix(Matrix* a);
void transposeMatrix(Matrix* a);
float MatrixMean(Matrix* a);
void sumMatrixRows(Matrix* row_indices, Matrix* matrix, Matrix* output);
void reshapeMatrix(Matrix* a,int rows,int columns);
void concatenateMatrix(Matrix* a,Matrix* b ,Matrix *c);
void zeroMatrix(Matrix *a);
void MatrixMultipy_bTa(Matrix* a, Matrix* b ,Matrix * c);
void scaleMatrix(Matrix* a,float scalar);
void clipMatrix(Matrix* a, float min, float max);
void randomizeMatrix(Matrix*a, float std_dv );
float get_min_element(Matrix* a);
float get_max_element(Matrix* a);

//helpers
float sumArray(float* v1, int size);
float dotProduct(float* v1, float* v2, int size);
#endif