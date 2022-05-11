#include "activation.h"
void relu(Matrix* unactivated, Matrix* activated)
{
    assert(checkDimension(unactivated , activated));
    for(int i=0 ; i< unactivated->rows * unactivated->columns ; i++)
    {
        activated->data[i] =  max( 0.0, unactivated->data[i]);
    }    
}
void clipped_relu(Matrix* unactivated, Matrix* activated)
{
    assert(checkDimension(unactivated , activated));
    for(int i=0 ; i< unactivated->rows * unactivated->columns ; i++)
    {
      activated->data[i] =  max(0.0, unactivated->data[i]);
      activated->data[i] =  min(1.0, activated->data[i]);
    }    
}
void sigmoid(Matrix* unactivated, Matrix* activated)
{
    assert(checkDimension(unactivated , activated));
    for(int i=0; i < unactivated-> rows * unactivated -> columns ; i++)
    {
        activated->data[i] = 1.0f / ( 1.0f + expf(-unactivated->data[i]));
    }  
}
void backprop_relu(Matrix* unactivated, Matrix* activated, Matrix* gradients)
{
    assert(checkDimension(unactivated, gradients) && checkDimension(unactivated, activated) );
    for(int i=0 ; i<unactivated->rows * unactivated->columns ; i++)
    {
        if( unactivated->data[i] < 0.0)
            gradients->data[i] = 0.0;
    }
}
void backprop_clipped_relu(Matrix* unactivated,Matrix* activated, Matrix* gradients)
{
    assert(checkDimension(unactivated, gradients) && checkDimension(unactivated, activated) );
    for(int i=0 ; i < unactivated->rows * unactivated->columns ; i++)
    {
        gradients->data[i] *= (unactivated->data[i] > 0.0 && unactivated->data[i] < 1.0);
    }
}
void backprop_sigmoid(Matrix* unactivated,Matrix* activated, Matrix* gradients)
{
    assert(checkDimension(unactivated, gradients) && checkDimension(unactivated, activated) ); 
    for(int i=0 ; i < unactivated->rows * unactivated->columns ; i++)
    {
        //derivative of sigmoid(x) = sigmoid(x)*(1 - sigmoid(x))
        gradients->data[i] *= (activated->data[i])*( 1 - activated->data[i]);
    }
}
