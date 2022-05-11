#ifndef _LOSS_H
#define _LOSS_H
#include "matrix.h"
typedef struct Loss
{
    float (*apply)(float ,float );
    float (*gradient)(float , float);
}Loss;

//loss functions
float mse(float prediction, float target);
float mae(float prediction, float target);
float mse2(float prediction, float target);

float gradient_mse(float prediction, float target);
float gradient_mae(float prediction, float target);
float gradient_mse2(float prediction, float target);

#endif