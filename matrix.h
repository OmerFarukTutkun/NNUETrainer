#ifndef _MATRIX
#define _MATRIX
#include <stdalign.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

//matrix with float entries
//rows are contigious in memory.
typedef struct Matrix{
    alignas(64) float* data; 
    int rows,columns; 
}Matrix;

Matrix* createMatrix(int rows ,int columns,float fillValue);
void freeMatrix(Matrix** a);
Matrix* MatrixFromArray(float* array,int rows,int columns );
Matrix* copyMatrix(Matrix* a);
void writeMatrix(Matrix*a,char* filename); 
Matrix* readMatrix(char* filename);
void printMatrix(Matrix* a);
void copyMatrixData(Matrix* a, Matrix *b);
int checkDimension(Matrix* a , Matrix* b);
int checkMemory(Matrix* a);

#endif