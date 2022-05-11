#ifndef _LAYER_H
#define _LAYER_H
#include "activation.h"

typedef struct LinearLayer
{
    Matrix* weights; // [out_size, in_size]
    Matrix* biases;  // [out_size, 1]

    Matrix* input; // [in_size, 1]
    Matrix* activated_output; //[out_size , 1]
    Matrix* unactivated_output;//[out_size , 1]

    Activation activation;

    Matrix* input_gradients; // [ input_size, 1] // gradients of the input of the layer
    Matrix* output_gradients;// [output_size, 1] gradients of the output of the layer
    Matrix* weight_gradients; // gradient of weights to be used in an optimizer
    Matrix* bias_gradients;

    int need_input_grad;

    void (*forward) (struct LinearLayer* self);
    void (*backward) (struct LinearLayer* self);
}LinearLayer;

void forwardLayer(LinearLayer* layer);

LinearLayer createLinearLayer(int input_size, int output_size, int need_input_grad, int is_sparse,Activation act); 
void freeLinearLayer(LinearLayer* layer);
void calculateWeightGradients(Matrix* weights_gradients, Matrix* input, Matrix* output_gradients);
void calculateWeightGradientsSparse(Matrix* weights_gradients, Matrix* input, Matrix* output_gradients);
void forwardLinearLayer(LinearLayer* layer);
void backwardLinearLayer(LinearLayer* layer);
void forward_Sparse(LinearLayer* layer);
void backwardSparse(LinearLayer* layer);
#endif