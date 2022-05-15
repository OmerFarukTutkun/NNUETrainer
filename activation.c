#include "activation.h"

//https://groups.google.com/g/comp.ai.neural-nets/c/gqekclNH3No
float fast_sigmoid(float x) {
    float absx = (float)fabs(x);
    float xx;
    if(absx>8.713655f) {
        if(x>0) 
            return 1.0f;
        else 
            return 0.0f;
    } 
    else {

        xx = x*x;
        if(absx>4.5f) {
            if(x>0)     
                return (float)(((3.2e-7*xx-8.544e-5)*xx+9.99869e-3)*x+0.953157);
            else 
                return (float)(((3.2e-7*xx-8.544e-5)*xx+9.99869e-3)*x+0.046843);
        } 
        else 
        {
            return (float)((((((-5e-8*xx+3.6e-6)*xx-1.0621e-4)*xx+1.75410e-3)*xx-0.02045660)*xx+0.24990936)*x+0.499985);
        }
    }
}


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
        activated->data[i] = fast_sigmoid(unactivated->data[i]);
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
