#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H
#include "nn.h"
#include "basic_operations.h"
typedef struct Velocity{
    Matrix* weights;
    Matrix* biases;
}Velocity;

typedef struct Moments{
    Matrix* moment1_W;
    Matrix* moment1_b;
    Matrix* moment2_W;
    Matrix* moment2_b;
}Moments;

typedef struct SGD{
    float lr;
    float momentum;
    int nesterov;
    float batch_size;
    Velocity velocity[100];
    NN* model;
}SGD;

typedef struct Adam{
    float lr;
    float batch_size;
    Moments moments[100];
    NN* model;
}Adam;

const float beta_1 = 0.9;
const float beta_2 = 0.999;
const float epsilon = 1e-8;


void initSGD(SGD* optimizer,NN* model, float lr ,float batch_size, float momentum)
{
    optimizer->lr = lr;
    optimizer->momentum = momentum;
    optimizer->model = model;
    optimizer->batch_size = batch_size;
    optimizer->nesterov = 0;
    for(int i=0; i<optimizer->model->num_of_layers ; i++)
    {
        optimizer->velocity[i].weights = createMatrix(optimizer->model->layers[i].weight_gradients->rows, optimizer->model->layers[i].weight_gradients->columns , 0.0f);
        optimizer->velocity[i].biases =  createMatrix(optimizer->model->layers[i].bias_gradients->rows, optimizer->model->layers[i].bias_gradients->columns , 0.0f);
    }
}
void optimizeSGD(SGD* optimizer)
{
    for(int i=0; i<optimizer->model->num_of_layers ; i++)
    {
        //calculate velocity = momentum*velocity - lr*gradients
        scaleMatrix( optimizer->velocity[i].weights ,   optimizer->momentum);
        scaleMatrix(optimizer->velocity[i].biases ,     optimizer->momentum);

        scaleMatrix( optimizer->model->layers[i].weight_gradients , -optimizer->lr/optimizer->batch_size);
        scaleMatrix(optimizer->model->layers[i].bias_gradients ,    -optimizer->lr/optimizer->batch_size);

        addMatrix(optimizer->velocity[i].weights,   optimizer->model->layers[i].weight_gradients,   optimizer->velocity[i].weights);
        addMatrix(optimizer->velocity[i].biases,    optimizer->model->layers[i].bias_gradients,     optimizer->velocity[i].biases);


        if(optimizer->nesterov)
        {
            addMatrix(optimizer->model->layers[i].weights,  optimizer->model->layers[i].weight_gradients,     optimizer->model->layers[i].weights);
            addMatrix(optimizer->model->layers[i].biases,   optimizer->model->layers[i].bias_gradients,      optimizer->model->layers[i].biases);
            scale_and_addMatrix(optimizer->model->layers[i].weights,  optimizer->velocity[i].weights,     optimizer->model->layers[i].weights , optimizer->momentum);
            scale_and_addMatrix(optimizer->model->layers[i].biases,   optimizer->velocity[i].biases,      optimizer->model->layers[i].biases ,  optimizer->momentum);
        }
        else
        {
            //update weights W = W + velocity
            addMatrix(optimizer->model->layers[i].weights,  optimizer->velocity[i].weights,     optimizer->model->layers[i].weights);
            addMatrix(optimizer->model->layers[i].biases,   optimizer->velocity[i].biases,      optimizer->model->layers[i].biases);
        }
        //zero gradients
        zeroMatrix(optimizer->model->layers[i].weight_gradients);
        zeroMatrix(optimizer->model->layers[i].bias_gradients);
    }
}
void freeSGD(SGD* optimizer)
{
    freeNN(optimizer->model);
    for(int i=0; i<optimizer->model->num_of_layers ; i++)
    {
        freeMatrix(&optimizer->velocity[i].weights);
        freeMatrix(&optimizer->velocity[i].biases);
    }
}
void  calculate_moment1(Matrix* moment, Matrix* gradient)
{
    for(int i=0; i < (moment->rows * moment->columns) ; i++)
    {
        moment->data[i] = ( beta_1 * moment->data[i] + (1-beta_1) * gradient->data[i] );
    }
}
void   calculate_moment2(Matrix* moment, Matrix* gradients)
{
    for(int i=0; i< moment->rows * moment->columns ; i++)
    {
        moment->data[i] = (beta_2 * moment->data[i] + (1-beta_2) * gradients->data[i] * gradients->data[i])  ;
    }
}
void  update_parameters_Adam(Matrix* moment1, Matrix* moment2, Matrix* parameters,float lr)
{
    for(int i=0; i<moment1->rows * moment1->columns ; i++)
    {
        parameters->data[i] -= lr*moment1->data[i] / ( sqrtf(moment2->data[i]) + epsilon) ;
    }
}

void initAdam(Adam* optimizer,NN* model, float lr ,float batch_size)
{
    optimizer->model = model;
    optimizer->lr = lr;
    optimizer->batch_size = batch_size;
    for(int i=0; i< optimizer->model->num_of_layers ; i++)
    {
        optimizer->moments[i].moment1_W = createMatrix(optimizer->model->layers[i].weight_gradients->rows, optimizer->model->layers[i].weight_gradients->columns , 0.0f);
        optimizer->moments[i].moment1_b =  createMatrix(optimizer->model->layers[i].bias_gradients->rows, optimizer->model->layers[i].bias_gradients->columns , 0.0f);
        optimizer->moments[i].moment2_W = createMatrix(optimizer->model->layers[i].weight_gradients->rows, optimizer->model->layers[i].weight_gradients->columns , 0.0f);
        optimizer->moments[i].moment2_b =  createMatrix(optimizer->model->layers[i].bias_gradients->rows, optimizer->model->layers[i].bias_gradients->columns , 0.0f);
    }

}
void  optimizeAdam(Adam* optimizer)
{
    for(int i=0; i< optimizer->model->num_of_layers ; i++)
    {
        //average the gradients
        scaleMatrix( optimizer->model->layers[i].weight_gradients ,  1.0/optimizer->batch_size);
        scaleMatrix(optimizer->model->layers[i].bias_gradients ,    1.0/optimizer->batch_size);

        // calculate moments 
        calculate_moment1(optimizer->moments[i].moment1_W, optimizer->model->layers[i].weight_gradients );
        calculate_moment1(optimizer->moments[i].moment1_b, optimizer->model->layers[i].bias_gradients);
        calculate_moment2(optimizer->moments[i].moment2_W, optimizer->model->layers[i].weight_gradients );
        calculate_moment2(optimizer->moments[i].moment2_b, optimizer->model->layers[i].bias_gradients );

        // update parameters
        update_parameters_Adam(optimizer->moments[i].moment1_W, optimizer->moments[i].moment2_W , optimizer->model->layers[i].weights, optimizer->lr);
        update_parameters_Adam(optimizer->moments[i].moment1_b, optimizer->moments[i].moment2_b,  optimizer->model->layers[i].biases,  optimizer->lr);
    }
}
void freeAdam(Adam* optimizer)
{
    freeNN(optimizer->model);
    for(int i=0; i<optimizer->model->num_of_layers ; i++)
    {
        freeMatrix(&optimizer->moments[i].moment1_W);
        freeMatrix(&optimizer->moments[i].moment1_b);
        freeMatrix(&optimizer->moments[i].moment2_W);
        freeMatrix(&optimizer->moments[i].moment2_b);
    }
}
#endif