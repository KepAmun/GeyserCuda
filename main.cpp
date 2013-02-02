
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <vector_types.h>

#include "VertexData.h"

#define USE_GPU 1


extern "C" void init_curand(curandState* state, unsigned long long seed);

extern "C" void launch_kernel(float3* pos, VertexData* data, int n, int time);


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// Particle positions for CPU
float3 pos[NUM_PARTICLES];

int elapsedTime = 0;
int lastTime = 0;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

VertexData * h_data;
VertexData* d_data;

void runCuda(struct cudaGraphicsResource **vbo_resource, VertexData* data)
{
    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource);

    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
    

    launch_kernel(dptr, data, NUM_PARTICLES, elapsedTime);
    

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void updateParticles()
{
    for(int i=0; i<NUM_PARTICLES; i++)
    {
        for(int n=0; n<200; n++){} // Padding work to compare GPU and CPU

        if( h_data->spawnTime[i] + h_data->lifespan[i] < elapsedTime )
        {
            pos[i].x = 0;
            pos[i].y = 0;
            pos[i].z = 0;
        
            float3 velocity;

            curandState localState = h_data->randState[i];
        
            float dTheta = h_data->dTheta * (rand()/(float)RAND_MAX);
            float dPhi = h_data->dPhi * (rand()/(float)RAND_MAX);
        
            float speed = (rand()/(float)RAND_MAX) * h_data->speedRange + h_data->speedMin;


            float theta = h_data->initTheta + dTheta;
            float phi = h_data->initPhi + dPhi;
            velocity.x = sin(phi) * sin(theta) * speed;
            velocity.y = cos(phi) * speed;
            velocity.z = sin(phi) * cos(theta) * speed;

            h_data->spawnTime[i] = elapsedTime;
            h_data->lifespan[i] = (rand()/(float)RAND_MAX) * h_data->lifespanRange + h_data->lifespanMin;

            h_data->randState[i] = localState;

            h_data->velocity[i] = velocity;
        }
        
        h_data->velocity[i].x += h_data->acceleration.x;
        h_data->velocity[i].y += h_data->acceleration.y;
        h_data->velocity[i].z += h_data->acceleration.z;

        pos[i].x += h_data->velocity[i].x;
        pos[i].y += h_data->velocity[i].y;
        pos[i].z += h_data->velocity[i].z;
    }
}


void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(34, timerEvent,0);
}


void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    
    elapsedTime = glutGet(GLUT_ELAPSED_TIME);
    int delta = elapsedTime-lastTime;
    lastTime = elapsedTime;

    printf("%f\n",1000.0f/delta);
    
    glColor3f(0.6, 0.80, 0.96);

    if(USE_GPU)
    {
        runCuda(&cuda_vbo_resource, d_data);

        // render from the vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
    else
    {
        updateParticles();

        glBegin(GL_POINTS);
        {
            for(int i=0; i<NUM_PARTICLES; i++)
            {
                glVertex3f(pos[i].x, pos[i].y, pos[i].z);
            }
        }
        glEnd();
    }
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBegin(GL_QUADS);
    {
        glColor3f(0.8,0.8,0.8);
        
        glVertex3f(1,1,1);
        glVertex3f(1,1,-1);
        glVertex3f(-1,1,-1);
        glVertex3f(-1,1,1);
        
        glVertex3f(1,-1,1);
        glVertex3f(1,-1,-1);
        glVertex3f(-1,-1,-1);
        glVertex3f(-1,-1,1);

        glVertex3f(1,1,1);
        glVertex3f(1,1,-1);
        glVertex3f(1,-1,-1);
        glVertex3f(1,-1,1);

        glVertex3f(-1,-1,-1);
        glVertex3f(-1,-1,1);
        glVertex3f(-1,1,1);
        glVertex3f(-1,1,-1);
    }
    glEnd();


    // Render x,y,z axes
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_LINES);
    {
        glColor3f(1,0,0); // x is red
        glVertex3f(0,0,0);
        glVertex3f(1,0,0);

        glColor3f(0,1,0); // y is blue
        glVertex3f(0,0,0);
        glVertex3f(0,1,0);

        glColor3f(0,0,1); // z is green
        glVertex3f(0,0,0);
        glVertex3f(0,0,1);
    }
    glEnd();
    glEnable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key)
    {
    case(27) :
        exit(0);
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


bool initGL(int argc, char** argv)
{
    int window_width = 640;
    int window_height = 480;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("GeyserCuda");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutTimerFunc(34, timerEvent,0);

    // default initialization
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1.0, 1000.0);
    
    glewInit();

    return true;
}


int main(int argc, char** argv)
{
    initGL(argc, argv);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES*3*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

    
    h_data = new VertexData();

    const float PI = 2 * acos(0.0f);
    h_data->PI = PI;
    const float PI_180 = PI/180;
    
    h_data->acceleration.x = 0;
    h_data->acceleration.y = -0.0008f;
    h_data->acceleration.z = 0;
    
    h_data->initTheta = 45 * PI_180;
    h_data->initPhi = 15 * PI_180;

    h_data->dTheta = 30 * PI_180;
    h_data->dPhi = 5 * PI_180;

    h_data->spread = 20 * PI_180;
    
    h_data->speedMin = 0.04f;
    h_data->speedRange = 0.005f;
    
    h_data->lifespanMin = 2000;
    h_data->lifespanRange = 2000;

    
    memset(h_data->spawnTime,-1,sizeof(int) * NUM_PARTICLES);
    memset(h_data->lifespan,0,sizeof(int) * NUM_PARTICLES);

    
    cudaMalloc((void**)&d_data, sizeof(VertexData));
    cudaMemcpy(d_data, h_data, sizeof(VertexData), cudaMemcpyHostToDevice);

    init_curand(d_data->randState,time(NULL));

    glutMainLoop();

    // delete VBO
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	
	vbo = 0;
    
    delete h_data;

    cudaFree(d_data);

    cudaDeviceReset();

    return 0;
}