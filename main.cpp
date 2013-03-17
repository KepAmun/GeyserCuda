
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <windows.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>

#include "VertexData.h"

#define USE_GPU 1


extern "C" void init_curand(curandState* state, unsigned long long seed);

extern "C" void launch_kernel(float3* pos, VertexData* data, int n, int now, int timeDelta);


// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// Particle positions for CPU
float3 pos[NUM_PARTICLES];

int elapsedTime = 0;
int lastTime = 0;
int numFrames = 0.0;
int lastFpsReset = 0;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

VertexData * h_data;
VertexData* d_data;


// Update particle positions on GPU 
void runCuda(struct cudaGraphicsResource **vbo_resource, VertexData* data, int now, int timeDelta)
{
    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource);

    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
    

    launch_kernel(dptr, data, NUM_PARTICLES, now, timeDelta);
    

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

// Update particle positions on CPU
void updateParticles(VertexData* data, int now, int timeDelta)
{
    for(int i=0; i<NUM_PARTICLES; i++)
    {

        if( data->spawnTime[i] + data->lifespan[i] < now )
        {
            pos[i].x = 0;
            pos[i].y = 0;
            pos[i].z = 0;
        
            float3 velocity;

            curandState localState = data->randState[i];
        
            float dTheta = data->dTheta * (rand()/(float)RAND_MAX);
            float dPhi = data->dPhi * (rand()/(float)RAND_MAX);
        
            float speed = (rand()/(float)RAND_MAX) * data->speedRange + data->speedMin;


            float theta = data->initTheta + dTheta;
            float phi = data->initPhi + dPhi;
            velocity.x = sin(phi) * sin(theta) * speed;
            velocity.y = cos(phi) * speed;
            velocity.z = sin(phi) * cos(theta) * speed;

            data->spawnTime[i] = elapsedTime;
            data->lifespan[i] = (rand()/(float)RAND_MAX) * data->lifespanRange + data->lifespanMin;

            data->randState[i] = localState;

            data->velocity[i] = velocity;
        }
        
        data->velocity[i].x += data->acceleration.x * timeDelta;
        data->velocity[i].y += data->acceleration.y * timeDelta;
        data->velocity[i].z += data->acceleration.z * timeDelta;

        pos[i].x += data->velocity[i].x * timeDelta;
        pos[i].y += data->velocity[i].y * timeDelta;
        pos[i].z += data->velocity[i].z * timeDelta;
    }
}


void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(1, timerEvent, 0); // Ask for 1000 frames per second.
}


void display()
{
    elapsedTime = glutGet(GLUT_ELAPSED_TIME);
    int timeDelta = elapsedTime-lastTime;

    numFrames++;

    // Update current frames per second only once each second
    if(lastTime/1000 < elapsedTime/1000)
    {
        printf("%f\n", numFrames/((elapsedTime-lastFpsReset)/1000.0f));
        numFrames = 0;
        lastFpsReset = elapsedTime;
    }

    lastTime = elapsedTime;


    // Init display and view angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    
    glColor3f(0.6, 0.80, 0.96); // Set color to light blue for particles

    if(USE_GPU)
    {
        runCuda(&cuda_vbo_resource, d_data, elapsedTime, timeDelta);

        // render from the vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
    else
    {
        updateParticles(h_data, elapsedTime, timeDelta);

        glBegin(GL_POINTS);
        {
            for(int i=0; i<NUM_PARTICLES; i++)
            {
                glVertex3f(pos[i].x, pos[i].y, pos[i].z);
            }
        }
        glEnd();
    }
    

    // Render framing box
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    
    glColor3f(0.8,0.8,0.8);

    // Top
    glBegin(GL_LINE_LOOP);
    {
        glVertex3f(1,1,1);
        glVertex3f(1,1,-1);
        glVertex3f(-1,1,-1);
        glVertex3f(-1,1,1);
    }
    glEnd();

    // Bottom
    glBegin(GL_LINE_LOOP);
    {   
        glVertex3f(1,-1,1);
        glVertex3f(1,-1,-1);
        glVertex3f(-1,-1,-1);
        glVertex3f(-1,-1,1);
    }
    glEnd();
    
    // Sides
    glBegin(GL_LINES);
    {
        glVertex3f(1,1,1);
        glVertex3f(1,-1,1);
        
        glVertex3f(1,1,-1);
        glVertex3f(1,-1,-1);

        glVertex3f(-1,1,1);
        glVertex3f(-1,-1,1);

        glVertex3f(-1,1,-1);
        glVertex3f(-1,-1,-1);

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

void setVSync(bool sync)
{	
	// Function pointer for the wgl extention function we need to enable/disable vsync
	typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALPROC)( int );
	PFNWGLSWAPINTERVALPROC wglSwapIntervalEXT = 0;

	const char *extensions = (char*)glGetString( GL_EXTENSIONS );

	if( strstr( extensions, "WGL_EXT_swap_control" ) != 0 )
	{
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALPROC)wglGetProcAddress( "wglSwapIntervalEXT" );

		if( wglSwapIntervalEXT )
			wglSwapIntervalEXT(sync);
	}
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
    
	glutTimerFunc(1, timerEvent,0); // Ask for 1000 frames per second.

    // default initialization
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1.0, 1000.0);

    setVSync(0);
    

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
    const float D_to_R = PI/180; // Degrees to Radians
    
    h_data->acceleration.x = 0;
    h_data->acceleration.y = -0.000002f;
    h_data->acceleration.z = 0;
    
    h_data->speedMin = 0.002f;
    h_data->speedRange = 0.00025f;
    
    h_data->initTheta = 45 * D_to_R;
    h_data->initPhi = 15 * D_to_R;

    h_data->dTheta = 30 * D_to_R;
    h_data->dPhi = 5 * D_to_R;

    h_data->spread = 20 * D_to_R;
    
    h_data->lifespanMin = 2000;
    h_data->lifespanRange = 2000;

    
    memset(h_data->spawnTime, -1, sizeof(int) * NUM_PARTICLES);
    memset(h_data->lifespan, 0, sizeof(int) * NUM_PARTICLES);

    
    cudaMalloc((void**)&d_data, sizeof(VertexData));
    cudaMemcpy(d_data, h_data, sizeof(VertexData), cudaMemcpyHostToDevice);

    init_curand(d_data->randState, time(NULL));

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