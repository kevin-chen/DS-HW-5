/*
    Jeff Epstein
    NYU Tandon, CS-UY 3254
    Conway's Life in CUDA
*/

/*
    Approach: Kernel, each thread takes 8 consecutive cells (since this is 
    how the cells char array is divided) and runs a for-loop over the 8 cells
    in its control and writes to the cells char array sequentially
*/

#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <unistd.h>
#include <time.h

/***********************
    Data structures
************************/

#define GRID_SIZE 512
#define CELL_SIZE 2
#define DELAY 10000


struct global {
    char *cells;
    char *cells_next;
    // TODO: Add any additional members to the global data structure
    //       required by your implementation
    char *gpu_cells;
    char *gpu_cells_next;
};

#ifdef CUDA
/***********************
    Game of Life, GPU version
************************/

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ bool get_cell(char *cells, int x, int y) {
    return (cells[(y*GRID_SIZE + x)/8] & (1<<(x%8))) != 0;
}

__device__ void set_cell_next(char *cells_next, int x, int y, bool val) {
    if (val)
        cells_next[(y*GRID_SIZE+x)/8] |= (1<<(x%8));
    else
        cells_next[(y*GRID_SIZE+x)/8] &= ~(1<<(x%8));
}

__device__ int count_neighbors(char *cells, int x, int y) {
    int count = 0;
    for (int i=x-1; i<=x+1; i++)
        for (int j=y-1; j<=y+1; j++)
            if (i!=x || j!=y)
                count += get_cell(cells, i,j); 
    return count;
}

__global__ void kernel(char *cells, char *cells_next) {
    // TODO: Provide implementation for kernel
    
    // 8 consective cells approach

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int startIndex = tid * 8;
    // printf("tid: %d, start: %d\n", tid, index);

    int index;
    for (int i = 0; i < 8; i++) {
        index = startIndex + i;
        int x = index / GRID_SIZE;
        int y = index % GRID_SIZE;
        // printf("x: %d, y: %d\n", x, y);

        int neighbors = count_neighbors(cells, x, y);
        bool newstate = 
            neighbors==3 || (get_cell(cells,x,y) && (neighbors == 2 || neighbors == 3));
        set_cell_next(cells_next,x,y,newstate);
    }

    // 2D Block approach

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;

    // if ( (i > 0 && i < GRID_SIZE - 1) && (j > 0 && j < GRID_SIZE - 1) ) {
    //     // int neighbors = count_neighbors(cells, i, j);
    //     // if (neighbors != 0)
    //     //     printf("Neighbors for %d, %d: %d\n", i, j, neighbors);

    //     int neighbors = count_neighbors(cells, i, j);
    //     bool newstate = 
    //         neighbors==3 || (get_cell(cells,i,j) && (neighbors == 2 || neighbors == 3));
    //     set_cell_next(cells_next,i,j,newstate);
    // }

}

void init_global(struct global *g) {
    // TODO: Initialize the global data structure as appropriate

    // Allocate the CPU arrays, cells & cells_next
    const int size = GRID_SIZE*GRID_SIZE/8;
    g->cells=(char*)malloc(size);
    g->cells_next=(char*)malloc(size);
    if (g->cells==NULL || g->cells_next==NULL) {
        fprintf(stderr, "Error: can't alloc data\n");
        exit(1);
    }

    // Initialize the CPU arrays
    for (int i=0; i<size; i++)
        g->cells[i]=0;

    // Allocate the GPU arrays
    HANDLE_ERROR( cudaMalloc( (void**)&g->gpu_cells, size * sizeof(char) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&g->gpu_cells_next, size * sizeof(char) ) );
}

bool get_cell(struct global *g, int x, int y) {
    // TODO: Provide implementation for get_cell function, used
    //       below to query state of automaton
    //       This can be similar to the CPU version
    return (g->cells[(y*GRID_SIZE + x)/8] & (1<<(x%8))) != 0;
}

void set_cell(struct global *g, int x, int y, bool val) {
    // TODO: Provide implementation for set_cell function, used
    //       below to load initial state of machine
    //       This can be similar to the CPU version
    if (val)
        g->cells[(y*GRID_SIZE+x)/8] |= (1<<(x%8));
    else
        g->cells[(y*GRID_SIZE+x)/8] &= ~(1<<(x%8));
}

void update(struct global *global) {
    // TODO: Implement the Conway's life algorithm on the GPU
    //       Feel free to refer to and copy code, as appropriate,
    //       from the CPU implementation below.

    const int size = GRID_SIZE * GRID_SIZE / 8;

    // Copy data from CPU array to GPU array
    HANDLE_ERROR( cudaMemcpy( global->gpu_cells, global->cells, size * sizeof(char),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( global->gpu_cells_next, global->cells_next, size * sizeof(char),
                              cudaMemcpyHostToDevice ) );

    // Call the kernel function to run one iteration on entire grid
    // printf("Invoking kernel");

    // 8 consective cells approach

    int numBlock = GRID_SIZE / 8;
    int numThread = GRID_SIZE;
    kernel<<<numBlock,numThread>>>(global->gpu_cells, global->gpu_cells_next);

    // 2D Block approach

    // dim3 threadsPerBlock(16, 16); 
    // dim3 numBlocks(GRID_SIZE / threadsPerBlock.x, GRID_SIZE / threadsPerBlock.y);
    // kernel<<<numBlocks,threadsPerBlock>>>(global->gpu_cells, global->gpu_cells_next);

    // Copy the calculcated data from GPU to CPU
    HANDLE_ERROR( cudaMemcpy( global->cells, global->gpu_cells, size * sizeof(char),
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( global->cells_next, global->gpu_cells_next, size * sizeof(char),
                              cudaMemcpyDeviceToHost ) );

    // Copy calculated data back to the cpu cells
    char *temp = global->cells;
    global->cells = global->cells_next;
    global->cells_next = temp;
}

#else

/***********************
    Game of Life, CPU version
************************/

/*
    Allocate memory for data structures
    and initialize data
*/
void init_global(struct global *g) {
    const int size = GRID_SIZE*GRID_SIZE/8;
    g->cells=(char*)malloc(size);
    g->cells_next=(char*)malloc(size);
    if (g->cells==NULL || g->cells_next==NULL) {
        fprintf(stderr, "Error: can't alloc data\n");
        exit(1);
    }
    for (int i=0; i<size; i++)
        g->cells[i]=0;
}

/*
    Returns true if a cell is alive at the given location
*/
bool get_cell(struct global *g, int x, int y) {
    return (g->cells[(y*GRID_SIZE + x)/8] & (1<<(x%8))) != 0;
}

void set_cell_next(struct global *g, int x, int y, bool val) {
    if (val)
        g->cells_next[(y*GRID_SIZE+x)/8] |= (1<<(x%8));
    else
        g->cells_next[(y*GRID_SIZE+x)/8] &= ~(1<<(x%8));
}

/*
    Set a cell alive or dead at the given location
*/
void set_cell(struct global *g, int x, int y, bool val) {
    if (val)
        g->cells[(y*GRID_SIZE+x)/8] |= (1<<(x%8));
    else
        g->cells[(y*GRID_SIZE+x)/8] &= ~(1<<(x%8));
}

/*
    Count neighbors of given cell
*/
int count_neighbors(struct global *g, int x, int y) {
    int count =0;
    for (int i=x-1; i<=x+1; i++)
        for (int j=y-1; j<=y+1; j++)
            if (i!=x || j!=y)
                count += get_cell(g,i,j); 
    return count;
}

/*
    Perform a complete step, storing the new state
    in global->cells
*/
void update(struct global *global) {
    for (int x=1; x<GRID_SIZE-1; x++)
        for (int y=1; y<GRID_SIZE-1; y++) {
            int neighbors = count_neighbors(global, x, y);
            bool newstate = 
                neighbors==3 || (get_cell(global,x,y) && (neighbors == 2 || neighbors == 3));
            set_cell_next(global,x,y,newstate);
        }    
    char *temp=global->cells;
    global->cells = global->cells_next;
    global->cells_next = temp;
}

#endif

/***********************
    X Window stuff
************************/

#define COLOR_RED "#FF0000"
#define COLOR_GREEN "#00FF00"
#define COLOR_BLACK "#000000"
#define COLOR_WHITE "#FFFFFF"

struct display
{
    Display         *display;
    Window          window;
    int             screen;
    Atom            delete_window;
    GC              gc;
    XColor          color1;
    XColor          color2;
    Colormap        colormap;
};

void init_display(struct display *dpy) {
        dpy->display = XOpenDisplay(NULL);
        if(dpy->display == NULL)
        {
            fprintf(stderr, "Error: could not open X dpy->display\n");
            exit(1);
        }
        dpy->screen = DefaultScreen(dpy->display);
        dpy->window = XCreateSimpleWindow(dpy->display, RootWindow(dpy->display, dpy->screen),
                0, 0, GRID_SIZE * CELL_SIZE, 
                GRID_SIZE * CELL_SIZE, 1,
                BlackPixel(dpy->display, dpy->screen), WhitePixel(dpy->display, dpy->screen));
        dpy->delete_window = XInternAtom(dpy->display, "WM_DELETE_WINDOW", 0);
        XSetWMProtocols(dpy->display, dpy->window, &dpy->delete_window, 1);
        XSelectInput(dpy->display, dpy->window, ExposureMask | KeyPressMask);
        XMapWindow(dpy->display, dpy->window);
        dpy->colormap = DefaultColormap(dpy->display, 0);
        dpy->gc = XCreateGC(dpy->display, dpy->window, 0, 0);
        XParseColor(dpy->display, dpy->colormap, COLOR_BLACK, &dpy->color1);
        XParseColor(dpy->display, dpy->colormap, COLOR_WHITE, &dpy->color2);
        XAllocColor(dpy->display, dpy->colormap, &dpy->color1);
        XAllocColor(dpy->display, dpy->colormap, &dpy->color2);

        XSelectInput(dpy->display,dpy->window, 
            KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask);

}

bool lookup_cell(struct global *g, int x, int y) {
    return (g->cells[(y*GRID_SIZE + x)/8] & (1<<(x%8))) != 0;
}


void do_display(struct global *global, struct display *dpy)
{
    XSetBackground(dpy->display, dpy->gc, dpy->color2.pixel);
    XClearWindow(dpy->display, dpy->window);

    for (int x=0; x<GRID_SIZE; x++)
        for (int y=0; y<GRID_SIZE; y++)
        {
            bool state = get_cell(global, x, y);
            if (state) {
                XSetForeground(dpy->display, dpy->gc, dpy->color1.pixel);
                XFillRectangle(dpy->display, dpy->window, dpy->gc, x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE);        
//              XDrawPoint(dpy->display, dpy->window, dpy->gc, x*CELL_SIZE, y*CELL_SIZE);
            }
        }

    XFlush(dpy->display);
}

void close_display(struct display *dpy)
{
    XDestroyWindow(dpy->display, dpy->window);
    XCloseDisplay(dpy->display);
}

/***********************
    Main program
************************/

void load_life(struct global *g, const char *fname) {
    char *line=NULL;
    size_t len = 0;
    ssize_t nread;
    int x,y;
    FILE *f = fopen(fname, "r");
    if (f==NULL) {
        fprintf(stderr,"Can't open file\n");
        exit(1);
    }
    while ((nread = getline(&line, &len, f)) != -1) {
        if (line[0]=='#')
            continue;
        if (nread<=1)
            continue;
        if (line[0]==13 || line[0]==10)
            continue;
        if (sscanf(line, "%d %d", &x, &y) != 2)
            continue;
        set_cell(g,x+GRID_SIZE/2,y+GRID_SIZE/2,1);
    }
    if (line)
        free(line);
    fclose(f);
}

void do_life(struct global *global) {
    bool running=1;
    struct display dpy;
    init_display(&dpy);
    while (running) {
        do_display(global, &dpy);
        usleep(DELAY);
        update(global);

        if (XPending(dpy.display)) {
            XEvent event;
            XNextEvent(dpy.display, &event);
            switch (event.type)
            {
                case ClientMessage:
                    if (event.xclient.data.l[0] == dpy.delete_window)
                        running=0;
                    break;
                case KeyPress:
                case ButtonPress:
                    running=0;
                    break;
                default:
                    break;
            }
        }
    }
    close_display(&dpy);
}

void perf_test(struct global *global) {
    int counter=10000;
    clock_t start = clock();
    clock_t diff;
    int msec;

    printf("Running performance test with %d iterations...\n", counter);
    fflush(stdout);

    while (counter>0) {
        update(global);
        counter--;
    }
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
}

int main(int argc, char *argv[]) {
    bool gui=1;
    struct global global;
    init_global(&global);

    #ifdef CUDA
        printf("Starting CUDA version of life....\n");
    #else
        printf("Starting CPU version of life....\n");
    #endif

    int argi;
    for (argi = 1; argi<argc; argi++)
        if (argv[argi][0]=='-' && argv[argi][1]=='i' && argv[argi][2]=='\0')
            gui=0;
        else
            break;

    if (argi==argc-1)
        load_life(&global, argv[argi]);
    else {
        fprintf(stderr,"Syntax: %s [-i] fname.lif\n", argv[0]);
        exit(1);
    }

    if (gui)
        do_life(&global);
    else
        perf_test(&global);
        
    return 0;
}