#include "basic_operations.h"
#include "omp.h"
//Loops are auto-vectorized 
void ElementwiseMultipyMatrix(Matrix* a, Matrix* b, Matrix* c)// elementwise matrix multipication: c = a .* b
{
    assert(checkDimension(a,b) && checkDimension(b,c));
    assert(checkMemory(a) && checkMemory(b) && checkMemory(c));
    for(int i=0 ; i < a->rows * a->columns ; i++)
        c->data[i] = a->data[i] * b->data[i];
}
void  addMatrix(Matrix* a, Matrix* b, Matrix* c)// matrix addition: c = a + b
{
    assert(checkDimension(a,b) && checkDimension(b,c));
    assert(checkMemory(a) && checkMemory(b) && checkMemory(c));
    for(int i=0 ; i < a->rows * a->columns ; i++)
        c->data[i] = a->data[i] + b->data[i];
}
void scale_and_addMatrix(Matrix*a, Matrix* b, Matrix* c, float x)
{
    assert(checkDimension(a,b) && checkDimension(b,c));
    assert(checkMemory(a) && checkMemory(b) && checkMemory(c));
    for(int i=0 ; i < a->rows * a->columns ; i++)
        c->data[i] = a->data[i] + x*b->data[i];
}
void subMatrix(Matrix* a, Matrix* b, Matrix* c)//matrix substraction: c = a - b
{
    assert(checkDimension(a,b) && checkDimension(b,c));
    assert(checkMemory(a) && checkMemory(b) && checkMemory(c));
    for(int i=0 ; i < a->rows * a->columns ; i++)
        c->data[i] = a->data[i] - b->data[i];
}
float traceMatrix(Matrix* a)
{
    assert(checkMemory(a));
    float result=0.0f;
    for(int i=0 ; i< a->rows ; i++)
        result += a->data[i*a->rows +i];
    return result; 
}
void randomizeMatrix(Matrix*a, float stddev )
{
    //https://github.com/AndyGrant/NNTrainer
    #define uniform()  ((double) (rand() + 1) / ((double) RAND_MAX + 2))
    #define random()   (sqrt(-2.0 * log(uniform())) * cos(2 * M_PI * uniform()))
    for (int j = 0; j < a->rows*a->columns; j++)
        a->data[j] = random() * stddev ;
    #undef uniform
    #undef random
}
void sumMatrixRows(Matrix* row_indices, Matrix* matrix, Matrix* output) 
{
    assert(checkMemory(row_indices));
    assert(checkMemory(matrix));
    assert(checkMemory(output));
    assert(matrix->columns == output->columns && output->rows == 1);
    Matrix addition;
    addition.rows = 1;
    addition.columns = matrix->columns;
    for(int i=0; i< row_indices->rows * row_indices->columns ; i++)
    {   
        addition.data = &matrix->data[ (int)row_indices->data[i] * matrix->columns];
        addMatrix(&addition, output,output);
    }
}
void MultipyMatrix_abT(Matrix* a, Matrix* b, Matrix* c) // C = A * B^T 
{
    assert(checkMemory(a) && checkMemory(b) && checkMemory(c));
    assert(c->rows == a->rows && c->columns == b->rows);
    for(int i=0 ; i<a->rows ; i++)
    {
        for(int j=0 ; j < b->rows ; j++)
        {
            c->data[i*b->rows + j] = dotProduct( &a->data[i*a->columns], &b->data[j*b->columns]  ,a->columns);
        }
    }
}
void MatrixMultipy_bTa(Matrix* a, Matrix* b ,Matrix * c)
{
    assert(checkMemory(a) &&  checkMemory(b) && checkMemory(c));
    assert(c->rows == b->columns && c->columns == a->columns);
    assert(a->rows == b->rows);
    assert(a->columns == 1);
    for(int j=0 ; j < b->columns; j++)
    {
            c->data[j] = a->data[0]*b->data[j];
    }
    for(int i=1 ; i < a->rows ; i++)
    {
        for(int j=0 ; j < b->columns; j++)
        {
                c->data[j] += a->data[i]*b->data[i*b->columns + j];
        }
    }
}

float MatrixMean(Matrix* a)
{
    return sumArray(a->data, a->rows*a->columns)/(a->rows*a->columns);
}
void zeroMatrix(Matrix *a)
{
    for(int i=0 ; i< a->rows * a->columns ; i++)
    {
        a->data[i] = 0;
    } 
}
void scaleMatrix(Matrix* a,float scalar)
{
    assert(checkMemory(a) );
    for(int i=0; i< a->rows * a->columns; i++)
        a->data[i] = a->data[i] *scalar;
}
void MultipyMatrix(Matrix* a, Matrix* b, Matrix* c) //C = A * B , a,b,c need to point different memory locations
{
    assert(c->rows == a->rows && c->columns == b->columns);
    assert(a->columns == b->rows);
    transposeMatrix(b); //transpose matrix so that columns are contigious in memory
    MultipyMatrix_abT(a,b,c);
    transposeMatrix(b); //transpose back
}
void transposeMatrix(Matrix* a)//transpose a matrix 
{
    assert(checkMemory(a) );
    if( a->rows > 1 &&  a->columns> 1)
    {
        float* transpose= (float*) _aligned_malloc(sizeof(float)* a->rows * a->columns , 64);
        for(int i=0; i < a->columns ; i++)
            for(int j=0; j < a->rows ; j++)
                transpose[i*a->rows + j] = a->data[j*a->columns + i];

        _aligned_free(a->data);
        a->data = transpose;
    }
    int temp = a->rows;
    a->rows = a->columns;
    a->columns = temp;
}
void reshapeMatrix(Matrix* a,int rows,int columns)
{
    assert(a->rows*a->columns == rows*columns && rows > 0 && columns > 0);
    a->rows = rows;
    a->columns = columns;
}
float sumArray(float* v1, int size)
{
    float result = 0.0f;
    for(int i=0 ; i < size ; i++)
        result += v1[i];
    return result;
}
float dotProduct(float* v1, float* v2, int size)
{
    float result = 0.0f;
    for(int i=0 ; i < size ; i++)
        result += v1[i] * v2[i];
    return result;
}
void concatenateMatrix(Matrix* a,Matrix* b ,Matrix *c)
{
    assert(a->columns == b->columns && a->columns == c->columns);
    assert(c->rows == (a->rows + b->rows));
    memcpy(c->data , a->data, sizeof(float) * a->rows*a->columns);
    memcpy( &c->data[a->rows* a->columns] , b->data, sizeof(float) * b->rows*b->columns);
}
void clipMatrix(Matrix* a, float min, float max)
{
     assert(checkMemory(a) );
     assert(max > min);
     for(int i=0 ; i<a->rows*a->columns; i++)
     {
         if(a->data[i] < min) 
            a->data[i] = min;
         else if(a->data[i] > max)
            a->data[i] = max;
     }
}
float get_max_element(Matrix* a)
{
    assert(checkMemory(a) );
    float max = a->data[0];
    for(int i=0; i<a->rows*a->columns ; i++)
    {
        max = a->data[i] > max ? a->data[i] : max;
    }
    return max;
}
float get_min_element(Matrix* a)
{
    assert(checkMemory(a) );
    float min = a->data[0];
    for(int i=0; i<a->rows*a->columns ; i++)
    {
        min = a->data[i] < min ? a->data[i] : min;
    }
    return min;
}