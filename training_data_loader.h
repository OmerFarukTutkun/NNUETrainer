#ifndef _DATA_LOADER_H
#define _DATA_LOADER_H
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "activation.h"
#define PAWN 1
#define KNIGHT 3
#define BISHOP 5
#define ROOK 7
#define QUEEN 9
#define B_PAWN 0
#define B_KNIGHT 2
#define B_BISHOP 4 
#define B_ROOK 6
#define B_QUEEN 8
#define Mirror(sq) ((7 - sq/8)*8 + sq%8)
#define SQ_MASK 63

const float scale = 275.0;
int pieces[2][32];      // pieces[0] for white perspective, pieces[1] for black
int weight_indices[2][32]; 
const uint8_t nn_indices[2][10] = { {5,0,6,1,7,2,8,3,9,4} ,{0,5,1,6,2,7,3,8,4,9}};

float score; // score of the position in wdl
int bit_cursor=0; // number of readed bits
uint8_t num =0 ;// number of pieces
int w_king,b_king;
uint8_t side;


int Horizontal_Mirror(int sq)
{ 
    return((sq/8)*8 + 7-sq%8);
}
int read_one_bit(uint8_t* data)
{
    int b= (data[bit_cursor/8] >> ( bit_cursor & 7)) & 1;
    bit_cursor++;
    return b;
}
int read_n_bit(uint8_t* data ,int n)
{
    int result = 0;
    for (int i = 0; i < n; ++i)
        result |= read_one_bit(data) ? (1 << i) : 0;

    return result;
}
uint8_t read_position(uint8_t* data)
{
    num = 0;
    bit_cursor =0;
    side = read_one_bit(data);
    uint16_t move = ((uint16_t)data[35] )<<8 | (uint16_t)data[34];
    uint8_t to = Horizontal_Mirror(move & SQ_MASK);
    w_king= Horizontal_Mirror(read_n_bit(data,6));
    b_king= Horizontal_Mirror(read_n_bit(data,6));
    for(int i=63 ; i>=0 ; i--)
    {
        if((i == w_king) || (i== b_king))
        {
            continue;
        }
        uint8_t piece = read_n_bit(data, 4);
        switch (piece)
        {
            case 1: case 5:
            case 3: case 7: case 9:

            if(i == to)//capture
                return 0;
            if(read_one_bit(data) )//black piece
                piece--;
            pieces[1][num] = nn_indices[1][piece]*64 + Mirror(i);
            pieces[0][num++] = nn_indices[0][piece]*64 + i;  break;
            default: bit_cursor -= 3; break;
        }
    }
    pieces[1][num] = 10*64+ Mirror(w_king);
    pieces[0][num++] = 10*64+ b_king;

    pieces[1][num] = 11*64+ Mirror(b_king);
    pieces[0][num++] = 11*64+ w_king;  
    b_king = Mirror(b_king);
    uint16_t t = ((uint16_t)data[33] )<<8 | (uint16_t)data[32];
    int16_t x =  (int16_t)t;
    score = fast_sigmoid(x/scale);
    for(int i=0; i<num; i++)
    {   
        weight_indices[0][i] = pieces[0][i];
        weight_indices[1][i] = pieces[1][i];
    }
    return num;
}
#endif