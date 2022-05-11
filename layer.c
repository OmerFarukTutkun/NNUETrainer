#include "layer.h"
#include "matrix.h"
#include "basic_operations.h"
#include "layer.h"
void calculateWeightGradients(Matrix* weights_gradients, Matrix* input, Matrix* output_gradients)
{
    for(int i=0; i< weights_gradients->rows; i++)
    {
        for(int j=0; j<weights_gradients->columns; j++)
        {
            weights_gradients->data[i*weights_gradients->columns + j] += output_gradients->data[i]*input->data[j];
        }
    }
}
void calculateWeightGradientsSparse(Matrix* weights_gradients, Matrix* input, Matrix* output_gradients)
{
    transposeMatrix(output_gradients);
    Matrix* addition = createMatrix(1, weights_gradients->columns, 0.0f);
    _aligned_free(addition->data);
    for(int i=0; i< input->columns* input->rows; i++)
    {
        addition->data = &weights_gradients->data[ (int)input->data[i] * weights_gradients->columns];
        addMatrix(addition, output_gradients, addition);
    }
    addition->data = NULL;
    transposeMatrix(output_gradients);
    freeMatrix(&addition);
}
void forwardLinearLayer(LinearLayer* layer)
{
    MultipyMatrix(layer->weights, layer->input, layer->unactivated_output);
    addMatrix(layer->biases, layer->unactivated_output, layer->unactivated_output);
    layer->activation.apply(layer->unactivated_output, layer->activated_output);
}
void backwardLinearLayer(LinearLayer* layer)
{
    layer->activation.backprop(layer->unactivated_output, layer->activated_output , layer->output_gradients);
    addMatrix(layer->output_gradients, layer->bias_gradients, layer->bias_gradients);
    if(layer->need_input_grad)
        MatrixMultipy_bTa(layer->output_gradients, layer->weights, layer->input_gradients);
    calculateWeightGradients(layer->weight_gradients, layer->input, layer->output_gradients);
}
void forward_Sparse(LinearLayer* layer)
{
    copyMatrixData(layer->unactivated_output,layer->biases );
    transposeMatrix(layer->unactivated_output);
    sumMatrixRows(layer->input, layer->weights, layer->unactivated_output);
    transposeMatrix(layer->unactivated_output);
    layer->activation.apply(layer->unactivated_output, layer->activated_output);
}
void backwardSparse(LinearLayer* layer)
{
    layer->activation.backprop(layer->unactivated_output, layer->activated_output , layer->output_gradients);
    addMatrix(layer->output_gradients, layer->bias_gradients, layer->bias_gradients);
    if(layer->need_input_grad)
        MultipyMatrix(layer->weight_gradients, layer->output_gradients, layer->input_gradients);
    calculateWeightGradientsSparse(layer->weight_gradients, layer->input, layer->output_gradients);
}
void freeLinearLayer(LinearLayer* layer)
{
    freeMatrix(&layer->weights);
    freeMatrix(&layer->biases);
    freeMatrix(&layer->input);
    freeMatrix(&layer->unactivated_output);
    freeMatrix(&layer->activated_output);
    freeMatrix(&layer->input_gradients);
    freeMatrix(&layer->output_gradients);
    freeMatrix(&layer->weight_gradients);
}
LinearLayer createLinearLayer(int input_size, int output_size, int need_input_grad, int is_sparse, Activation act)
{
    LinearLayer layer;
    if(is_sparse)//for the first layers of  NNs
    {
        layer.weights = createMatrix(input_size, output_size, 0.0);
        layer.weight_gradients = createMatrix(input_size, output_size, 0.0);
        layer.forward = &forward_Sparse;
        layer.backward= &backwardSparse;
        layer.input = NULL;
    }
    else
    {
        layer.weights = createMatrix(output_size, input_size, 0.0);
        layer.weight_gradients = createMatrix(output_size, input_size, 0.0);
        layer.forward = &forwardLinearLayer;
        layer.backward= &backwardLinearLayer;
        layer.input = createMatrix(input_size, 1, 0.0);
    }

    layer.biases = createMatrix(output_size, 1, 0.0);
    layer.bias_gradients = createMatrix(output_size, 1, 0.0);
    randomizeMatrix(layer.weights, sqrt(2.0/(input_size)));
    layer.need_input_grad = need_input_grad;
    layer.unactivated_output = createMatrix(output_size, 1, 0.0);
    layer.activated_output = createMatrix(output_size, 1, 0.0);

    if(need_input_grad)
        layer.input_gradients = createMatrix(input_size, 1, 0.0);
    else
        layer.input_gradients = NULL;
    
    layer.output_gradients = createMatrix(output_size, 1, 0.0);
    layer.activation = act;
    return layer;
}
