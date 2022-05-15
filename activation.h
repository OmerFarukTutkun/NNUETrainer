#ifndef _ACTIVATION_H
#define _ACTIVATION_H
#include "matrix.h"
typedef struct Activation
{
    void (*apply)(Matrix* ,Matrix*);
    void (*backprop)(Matrix*, Matrix*, Matrix*);
}Activation;

float fast_sigmoid(float x);


//activation functions
void relu(Matrix* unactivated, Matrix* activated);
void sigmoid(Matrix* unactivated, Matrix* activated);
void clipped_relu(Matrix* unactivated, Matrix* activated);

//multipy gradient vector by gradients of activation functions in backpropagation
void backprop_relu(Matrix* unactivated, Matrix* activated, Matrix* gradients);
void backprop_sigmoid(Matrix* unactivated, Matrix* activated, Matrix* gradients);
void backprop_clipped_relu(Matrix* unactivated, Matrix* activated, Matrix* gradients);

#endif