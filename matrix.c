#include "matrix.h"

Matrix* createMatrix(int rows ,int columns, float fillValue) //create mxn matrix 
{
    assert(rows > 0 && columns > 0);
    float * data = (float* ) _aligned_malloc(rows*columns*sizeof(float),64 ); //64-bit aligned float data
    Matrix* matrix = (Matrix *)_aligned_malloc(sizeof(Matrix),64 ); 
    assert(matrix != NULL && data != NULL);
    for(int i=0 ; i < (rows *columns) ; i++)
        data[i] = fillValue;
    *matrix = (Matrix) {.data = data, .rows =rows, .columns =columns};
    return matrix;
}
void freeMatrix(Matrix** matrix)
{
    if(*matrix == NULL)
        return ;
    if( (*matrix)->data != NULL)
        _aligned_free((*matrix)->data);

    (*matrix)->data = NULL;
    _aligned_free(*matrix); 
    *matrix = NULL;
}
void printMatrix(Matrix* matrix)
{
    assert(checkMemory(matrix));
    printf("Matrix with size %d x %d\n", matrix->rows , matrix->columns);
    if(matrix->rows >40 || matrix->columns >40)
    {
        printf("Matrix is too big to print\n");
        return ;
    }
    for(int i=0; i < matrix->rows ; i++)
    {
        for(int j=0; j < matrix->columns; j++)
        {
            printf("%10.5f  ", matrix->data[i* matrix->columns + j]);
        }
        printf("\n");
    }
}
void writeMatrix(Matrix* a , char* filename)
{
    FILE* file = fopen(filename , "wb");
    if(file == NULL)
    {
        printf("File could not open\n");
        return ;
    }
    fwrite(&a->rows, sizeof(int) , 1, file);
    fwrite(&a->columns, sizeof(int) , 1, file);
    fwrite(a->data, sizeof(float) , a->rows*a->columns, file);
    fclose(file);
}
Matrix* readMatrix(char* filename)
{
    FILE* file = fopen(filename , "rb");
    if(file == NULL)
    {
        printf("File could not open\n");
        return 0;
    }
    int sizes[2];
    fread(sizes, sizeof(int) , 2, file);
    printf("Mtarix with row:%d colummn: %d\n" , sizes[0] , sizes[1]);
    Matrix * matrix = createMatrix(sizes[0] , sizes[1], 0.0f);
    fread(matrix->data, sizeof(float) , sizes[0]*sizes[1], file);
    fclose(file);
    return matrix;
}
Matrix* copyMatrix(Matrix* a)
{
    assert(checkMemory(a));
    Matrix * b = (Matrix*) _aligned_malloc(sizeof(Matrix) , 64);
    float* copy= (float*) _aligned_malloc(sizeof(float)* a->rows * a->columns , 64);
    memcpy(copy, a->data,sizeof(float)* a->rows * a->columns );
    *b = (Matrix) { .data=copy , .rows = a->rows , .columns= a->columns};
    return b;
}
void copyMatrixData(Matrix* a, Matrix *b) 
{
    assert(checkDimension(a,b));
    assert(checkMemory(a) && checkMemory(b));
    memcpy(a->data, b->data,  sizeof(float)* a->rows * a->columns );
}
Matrix* MatrixFromArray(float* array,int rows,int columns )
{
    Matrix* a = createMatrix(rows,columns,0.0f);
    memcpy(a->data , array , sizeof(float)* rows * columns);
    return a;
}
int checkDimension(Matrix* a, Matrix* b)
{
    return (a->rows == b->rows && a->columns == b->columns);
}
int checkMemory(Matrix* a)
{
    return !( a == NULL || a->data == NULL);
}