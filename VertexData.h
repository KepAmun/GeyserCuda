
#include <vector_types.h>
#include <curand_kernel.h>

#define NUM_PARTICLES (768*256)

struct VertexData
{
    float PI;

    float3 acceleration;

    float3 velocity[NUM_PARTICLES];

    float speedMin;
    float speedRange;
    
    float initTheta;
    float initPhi;
    
    float dTheta;
    float dPhi;

    float spread;

    float lifespanMin;
    float lifespanRange;
    
    int spawnTime[NUM_PARTICLES]; // in ms since app launch
    int lifespan[NUM_PARTICLES]; // in ms
    curandState randState[NUM_PARTICLES];
};
