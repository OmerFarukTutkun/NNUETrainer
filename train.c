#include "matrix.h"
#include <dos.h>
#include <time.h>
#include "loss.h"
#include "optimizer.h"
#include <stdint.h>
#include "training_data_loader.h"

#define NUMBER_OF_SAMPLE 10000000
#define EPOCH_SIZE 100000000
#define TRANINNG_FILE "D:\\Desktop\\SATRANC\\Stockfish\\farseer_shuffled.bin"
#define BATCH_SIZE 16384
#define LR 0.002

int main()
{
    srand(time(NULL));

    NN model;
    initNN(&model);
    Adam optimizer;
    initAdam(&optimizer,&model,LR, BATCH_SIZE);

    Matrix* input1 = createMatrix(32, 1,0.0);  
    Matrix* input2 = createMatrix(32, 1,0.0);  
    model.layers[1].input = input2;
    model.layers[0].input = input1;

    float prediction,loss,total_loss=0.0,metric=0.0;
    long long unsigned int m=0,e=0;
    time_t previous_time = clock();
    uint8_t* buffer =(uint8_t*)malloc( 40*NUMBER_OF_SAMPLE); 

    FILE* file = fopen(TRANINNG_FILE , "rb");
    if(file == NULL)
    {
        printf("Traning data could not open\n");
        return 0;
    } 

    for(int epoch=1; epoch <=100; epoch++)
    {
        printf("started epoch %d\n", epoch);
        for(int k=0 ; k< EPOCH_SIZE /NUMBER_OF_SAMPLE ; k++ )
        {
            fread(buffer , sizeof(uint8_t) , 40*NUMBER_OF_SAMPLE ,file);

            m=1;
            total_loss=0;
            e=0;
            previous_time = clock();
            metric = 0.0;

            for(int i=0 ; i<NUMBER_OF_SAMPLE ; i++)
            {
                if(m  >= NUMBER_OF_SAMPLE)
                    break;
                if(!read_position(&buffer[ 40*m ] ))
                {
                    m++;
                    continue;
                }
                m ++;
                e++;
                for(int j=0 ; j<num; j++)
                {
                    input1->data[j] = 0.001 + (float)active_neurons[side][j]; 
                    input2->data[j] = 0.001 + (float)active_neurons[!side][j]; 
                }
                input2->rows= num;
                input1->rows= num;
                model.forward(&model);

                prediction = model.layers[model.num_of_layers -1].activated_output->data[0];
                total_loss +=  model.loss.apply(prediction, score);
                metric += mae(prediction,score);

                model.layers[model.num_of_layers -1].output_gradients->data[0] = model.loss.gradient(prediction, score);
                model.backward(&model);

                if(e % BATCH_SIZE == 0)
                {
                    optimizeAdam(&optimizer);       
                }
            }
            printf("loss : %.4f MAE: %.4f ", (float)total_loss/(float)e ,(float)metric/(float)e);
            printf("data/second: %ld\n", e/ ((clock() - previous_time)/CLOCKS_PER_SEC) );
        }
        char filename[100];
        time_t t;
        t = time(NULL);
        struct tm tm = *localtime(&t);
        sprintf(filename, "network_%d.%d.%d_%d.%d_epoch%d.bin",tm.tm_mday, tm.tm_mon+1, tm.tm_year+1900,tm.tm_hour,tm.tm_min, epoch );
        saveNN(&model , filename);
    }
    free(buffer);
    fclose(file);
    return 0;
}