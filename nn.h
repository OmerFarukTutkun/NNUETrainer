#ifndef _NN_H
#define _NN_H
#include "matrix.h"
#include "basic_operations.h"
#include "layer.h"
#include "loss.h"
#define WEIGHT_SCALE 512.0
int INPUT_SIZE = 768;
int L1= 128;
int L2= 8;
int L3 =32;
typedef struct NN{
    LinearLayer layers[100];
    Loss loss;
    int num_of_layers;
    void (*forward)(struct NN* self);
    void (*backward)(struct NN* self);
}NN;
void forward_model(NN* model)
{
    for(int i=0 ; i<model->num_of_layers ; i++)
        model->layers[i].forward(&model->layers[i]);
}
void backward_model(NN* model)
{
    for(int i= model->num_of_layers -1; i >= 0; i--)
        model->layers[i].backward(&model->layers[i]);
}
void initNN(NN* model)
{
    model->layers[0] = createLinearLayer(INPUT_SIZE,L1, FALSE,TRUE , (Activation){clipped_relu, backprop_clipped_relu}); // us
    model->layers[1] = createLinearLayer(INPUT_SIZE,L1, FALSE,TRUE , (Activation){clipped_relu, backprop_clipped_relu}); // enemy
    model->layers[2] = createLinearLayer(2*L1,L2, TRUE,FALSE , (Activation){clipped_relu, backprop_clipped_relu}); 
    model->layers[3] = createLinearLayer(L2,L3, TRUE,FALSE , (Activation){clipped_relu, backprop_clipped_relu});
    model->layers[4] = createLinearLayer(L3,1, TRUE,FALSE , (Activation){sigmoid, backprop_sigmoid});

    model->layers[1].is_trainable= FALSE;


    model->layers[1].weights            = model->layers[0].weights           ;
    model->layers[1].bias_gradients     = model->layers[0].bias_gradients    ;
    model->layers[1].biases             = model->layers[0].biases            ;
    model->layers[1].weight_gradients   = model->layers[0].weight_gradients  ;
    //Make forward connections
    model->layers[0].activated_output->data =  model->layers[2].input->data ; 
    model->layers[1].activated_output->data = &model->layers[2].input->data[L1] ; 
    model->layers[2].activated_output->data =  model->layers[3].input->data ; 
    model->layers[3].activated_output->data =  model->layers[4].input->data ;

    //backward connections
    model->layers[0].output_gradients->data =  model->layers[ 2].input_gradients->data ;
    model->layers[1].output_gradients->data = &model->layers[ 2].input_gradients->data[L1] ;
    model->layers[2].output_gradients->data =  model->layers[ 3].input_gradients->data ;
    model->layers[3].output_gradients->data =  model->layers[ 4].input_gradients->data ;

    model->num_of_layers =5;
    Loss loss= (Loss) { .apply = mse, .gradient =gradient_mse };
    model->loss = loss;
    model->forward = forward_model;
    model->backward = backward_model;
}

void saveNN(NN* model,char* filename)
{
    FILE* file = fopen(filename,"wb");
    int16_t* feature_weights = (int16_t*)malloc(INPUT_SIZE*L1*sizeof(int16_t));
    int16_t feature_biases[L1];
    int16_t weights1[2*L1*L2];
    int32_t biases1[L2];


    for(int i=0; i<INPUT_SIZE*L1 ; i++)
        feature_weights[i] =(int16_t)(WEIGHT_SCALE*model->layers[0].weights->data[i]);
    for(int i=0; i<L1 ; i++)
        feature_biases[i] =(int16_t)(WEIGHT_SCALE*model->layers[0].biases->data[i]);

    fwrite(feature_weights, sizeof(int16_t), INPUT_SIZE*L1,file);
    fwrite(feature_biases, sizeof(int16_t), L1,file);

    transposeMatrix(model->layers[3].weights);
    for(int i=0; i<2*L1*L2 ; i++)
        weights1[i] =(int16_t)(WEIGHT_SCALE*model->layers[2].weights->data[i]);
    for(int i=0; i<L2 ; i++)
        biases1[i] =(int32_t)(WEIGHT_SCALE*WEIGHT_SCALE*model->layers[2].biases->data[i]);

    fwrite(weights1, sizeof(int16_t),2*L1*L2,file);
    fwrite(biases1, sizeof(int32_t), L2,file);
    fwrite(model->layers[3].weights->data, sizeof(float), L2*L3,file);
    fwrite(model->layers[3].biases->data, sizeof(float), L3,file);
    fwrite(model->layers[4].weights->data, sizeof(float), L3,file);
    fwrite(model->layers[4].biases->data, sizeof(float), 1,file);
    transposeMatrix(model->layers[3].weights);


    fclose(file);
    free(feature_weights);
}
void readNN(NN* model,char* filename)
{
     FILE* file = fopen(filename,"rb");
   int16_t* feature_weights = (int16_t*)malloc(INPUT_SIZE*L1*sizeof(int16_t));
    int16_t feature_biases[L1];
    int16_t weights1[2*L1*L2];
    int32_t biases1[L2];

    fread(feature_weights, sizeof(int16_t), INPUT_SIZE*L1,file);
    fread(feature_biases, sizeof(int16_t), L1,file);

    for(int i=0; i<INPUT_SIZE*L1 ; i++)
        model->layers[0].weights->data[i] = feature_weights[i]/WEIGHT_SCALE;
    for(int i=0; i<L1 ; i++)
        model->layers[0].biases->data[i] = feature_biases[i]/WEIGHT_SCALE;

    transposeMatrix(model->layers[3].weights);
    fread(weights1, sizeof(int16_t), 2*L1*L2,file);
    fread(biases1, sizeof(int32_t), L2,file);
    fread(model->layers[3].weights->data, sizeof(float), L2*L3,file);
    fread(model->layers[3].biases->data, sizeof(float), L3,file);
    fread(model->layers[4].weights->data, sizeof(float), L3,file);
    fread(model->layers[4].biases->data, sizeof(float), 1,file);

    transposeMatrix(model->layers[3].weights);

    for(int i=0; i<2*L1*L2 ; i++)
        model->layers[2].weights->data[i] = weights1[i]/WEIGHT_SCALE;
    for(int i=0; i<L2 ; i++)
        model->layers[2].biases->data[i] = biases1[i]/(WEIGHT_SCALE*WEIGHT_SCALE);
    fclose(file);
    free(feature_weights);
}

//does not work for now
void freeNN(NN* model)
{
    for(int i=0; i< model->num_of_layers; i++)
        freeLinearLayer(&model->layers[i]);
}
#endif