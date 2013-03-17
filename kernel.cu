
#include <math.h>
#include <curand_kernel.h>

#include "VertexData.h"

#ifndef _KERNEL_H_
#define _KERNEL_H_


__global__ void init_curand_kernel(curandState* state, unsigned long long seed)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    curand_init ( seed, i, 0, &state[i] );

}


__global__ void kernel(float3* pos, VertexData* data, int n, int now, int timeDelta)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    

    float3 p = pos[i];
    
    if( data->spawnTime[i] + data->lifespan[i] < now )
    {
        p.x = 0;
        p.y = 0;
        p.z = 0;
        
        float3 velocity;

        curandState localState = data->randState[i];
        
        float dTheta = data->dTheta * curand_normal(&localState)/2;
        float dPhi = data->dPhi * curand_normal(&localState)/2;
        
        float speed = curand_normal(&localState)/2 * data->speedRange + data->speedMin;


        float theta = data->initTheta + dTheta;
        float phi = data->initPhi + dPhi;
        velocity.x = __sinf(phi) * __sinf(theta) * speed;
        velocity.y = __cosf(phi) * speed;
        velocity.z = __sinf(phi) * __cosf(theta) * speed;

        data->spawnTime[i] = now;
        data->lifespan[i] = curand_normal(&localState)/2 * data->lifespanRange + data->lifespanMin;

        data->randState[i] = localState;

        data->velocity[i] = velocity;
    }
    
    data->velocity[i].x += data->acceleration.x * timeDelta;
    data->velocity[i].y += data->acceleration.y * timeDelta;
    data->velocity[i].z += data->acceleration.z * timeDelta;
    
    p.x += data->velocity[i].x * timeDelta;
    p.y += data->velocity[i].y * timeDelta;
    p.z += data->velocity[i].z * timeDelta;

    pos[i] = p;
}

extern "C" void init_curand(curandState* state, unsigned long long seed)
{
    // execute the kernel
    dim3 grid(GRID_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    init_curand_kernel<<< grid, block >>>(state, seed);
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float3* pos, VertexData* data, int n, int now, int timeDelta)
{
    // execute the kernel
    dim3 grid(GRID_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    kernel<<< grid, block >>>(pos, data, n, now, timeDelta);
}

#endif // #ifndef _KERNEL_H_
