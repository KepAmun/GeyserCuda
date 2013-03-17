
#include <curand_kernel.h>

const int GRID_SIZE = 256;
const int BLOCK_SIZE = 768;
const int NUM_PARTICLES = GRID_SIZE * BLOCK_SIZE;

struct VertexData
{
    float PI;

    float3 acceleration;

    float3 velocity[NUM_PARTICLES];

    float speedMin; // in units per millisecond
    float speedRange;
    
    float initTheta;
    float initPhi;
    
    float dTheta;
    float dPhi;

    float spread;

    float lifespanMin;
    float lifespanRange;
    
    int spawnTime[NUM_PARTICLES]; // in milliseconds since app launch
    int lifespan[NUM_PARTICLES]; // in milliseconds

    curandState randState[NUM_PARTICLES];
};
