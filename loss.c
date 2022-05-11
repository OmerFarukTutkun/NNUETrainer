#include "loss.h"
#include <math.h>
float mse(float prediction, float target)
{
    return (target - prediction)*(target - prediction);
}
float mae(float prediction, float target)
{
    return (target - prediction) < 0.0 ? (prediction - target) : (target - prediction);
}
float gradient_mse(float prediction, float target)
{
    //take derivative with respect to prediction
    return -2*(target - prediction);
}
float gradient_mae(float prediction, float target)
{
    return target > prediction ? -1.0 : 1.0;
}
float mse2(float prediction, float target)
{
    return powf(fabs(target - prediction) , 2.6);
}
float gradient_mse2(float prediction, float target)
{
    return powf(fabs(target - prediction) , 1.6)*(target > prediction ? -2.6 : 2.6);
}