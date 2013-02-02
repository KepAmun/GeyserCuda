#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include "VertexData.h"

#ifndef _KERNEL_H_
#define _KERNEL_H_


__global__ void init_curand_kernel(curandState* state, unsigned long long seed)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    curand_init ( seed, i, 0, &state[i] );

}


__global__ void kernel(float3* pos, VertexData* data, int n, int now)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    for(int n=0; n < 200; n++){} // Padding work to compare GPU and CPU

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
    
    data->velocity[i].x += data->acceleration.x;
    data->velocity[i].y += data->acceleration.y;
    data->velocity[i].z += data->acceleration.z;
    
    p.x += data->velocity[i].x;
    p.y += data->velocity[i].y;
    p.z += data->velocity[i].z;

    pos[i] = p;
}

extern "C" void init_curand(curandState* state, unsigned long long seed)
{
    // execute the kernel
    dim3 grid(256, 1, 1);
    dim3 block(768, 1, 1);
    init_curand_kernel<<< grid, block >>>(state, seed);
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float3* pos, VertexData* data, int n, int time)
{
    // execute the kernel
    dim3 grid(256, 1, 1);
    dim3 block(768, 1, 1);
    kernel<<< grid, block >>>(pos, data, n, time);
}

#endif // #ifndef _KERNEL_H_
