#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define dtype float
#define DTYPE_MAX FLT_MAX
#define FULL_MASK 0xffffffff

#define M 30
#define N 40
#define DIMS 128

#define DISTS_WIDTH 1 

#define MAX_POINTS 10000
#define MAX_DIMS 1024


#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void computeDists(int m, int n, int dims, dtype *s1, dtype *s2, dtype *res) {
    int tid = threadIdx.x;    
    int wid = tid / 32;
    int warpsPerBlock = (blockDim.x + 31) / 32; 
    
    int start_i = blockIdx.x * DISTS_WIDTH;
    int end_i = min(start_i + DISTS_WIDTH, m);

    for (int i = start_i; i < end_i; i++) {
        // March the warps along s2
        for (int j = wid; j < n; j += warpsPerBlock) {    

            // Compute the sum
            dtype sum = 0;
            for (int k = tid % 32; k < dims; k += 32) {
                dtype diff = s1[i*dims + k] - s2[j*dims + k];
                sum += diff * diff;
            }
            
            // Warp level reduction
            sum += __shfl_down_sync(FULL_MASK, sum, 16);
            sum += __shfl_down_sync(FULL_MASK, sum, 8);
            sum += __shfl_down_sync(FULL_MASK, sum, 4);
            sum += __shfl_down_sync(FULL_MASK, sum, 2);
            sum += __shfl_down_sync(FULL_MASK, sum, 1); 

            if (tid % 32 == 0) res[i*n + j] = sqrt(sum);
        }
    }
}

// Computes the minimum of each row (i.e. minimum distance of each point in S1
// over all points in S2)
// TODO: Break this down into two parts to handle m > 1024
//
// dists: `m * n` array of distances
__global__ void minReduceEachRow(int m, int n, int dims, dtype *dists) {
    __shared__ dtype vec[32];

    int row = blockIdx.x;
    int i = threadIdx.x;
    dtype v = DTYPE_MAX;

    if (i < n) v = dists[row*n + i]; 
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2)); 
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1)); 
    if (i % 32 == 0) vec[i>>5] = v;

    __syncthreads();  
    if (i > 31) return;

    v = DTYPE_MAX;
    if (i < (n >> 5)) v = vec[i];
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1));
    if (i == 0) dists[row*n] = v;
}

// Compute the minimum of each column (i.e. minimum distance of each point in S2
// over all points in S1)
// TODO: Break this down into two parts to handle n > 1024
//
// dists: `m * n` array of distances
__global__ void minReduceEachCol(int m, int n, int dims, dtype *dists) {
    __shared__ dtype vec[32];

    int col = blockIdx.x;
    int i = threadIdx.x;
    dtype v = DTYPE_MAX;

    if (i < m) v = dists[i*m + col];
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1));
    if (i % 32 == 0) vec[i>>5] = v;

    __syncthreads();
    if (i > 31) return;

    v = DTYPE_MAX;
    if (i < (n >> 5)) v = vec[i];
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1));
    if (i == 0) dists[col] = v;
}

// Computes the sum of first elements of each row
//
// dists: `m * n` array with first element of each row being the minimum element
__global__ void sumRowMins(int m, int n, int dims, dtype *dists) {
    __shared__ dtype vec[32];

    int i = threadIdx.x;
    dtype v = 0;

    if (i < m) v = dists[i*n];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    if (i % 32 == 0) vec[i>>5] = v;
    
    __syncthreads();
    if (i > 31) return;

    v = (i < m>>5) * vec[i];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    if (i == 0) dists[0] = v;
}

// Computes the sum of first elements of each column
//
// dists: `m * n` array with first element of each column begin the minimum element
__global__ void sumColMins(int m, int n, int dims, dtype *dists) { 
    __shared__ dtype vec[32];

    int i = threadIdx.x;
    dtype v = 0;

    if (i < n) v = dists[i];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    if (i % 32 == 0) vec[i>>5] = v;
    
    __syncthreads();
    if (i > 31) return;

    v = (i < n>>5) * vec[i];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    if (i == 0) dists[0] = v;
}

// Implementation of Chamfer distance calculation with verbose parameter added
dtype computeChamferDistance(dtype *s1_h, int m, dtype *s2_h, int n, int dims, int verbose) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    if (verbose) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    dtype *s1_d, *s2_d, *dists_d, *dists2_d;    

    gpuErrchk(cudaMalloc(&s1_d, m * dims * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&s2_d, n * dims * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&dists_d, m * n * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&dists2_d, m * n * sizeof(dtype)));

    gpuErrchk(cudaMemcpy(s1_d, s1_h, m * dims * sizeof(dtype), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(s2_d, s2_h, n * dims * sizeof(dtype), cudaMemcpyHostToDevice));

    // Compute distances 
    computeDists<<<(n + DISTS_WIDTH - 1) / DISTS_WIDTH, 1024>>>(m, n, dims, s1_d, s2_d, dists_d);
    
    // Copy distances
    gpuErrchk(cudaMemcpy(dists2_d, dists_d, m * n * sizeof(dtype), cudaMemcpyDeviceToDevice));

    // Reductions along the two dimensions can be done parallely on different streams
    minReduceEachRow<<<m, (((n-1)/32)+1)*32, 0>>>(m, n, dims, dists_d);
    minReduceEachCol<<<n, (((m-1)/32)+1)*32, 0>>>(m, n, dims, dists2_d); 

    sumRowMins<<<1, m, 0>>>(m, n, dims, dists_d);
    sumColMins<<<1, n, 0>>>(m, n, dims, dists2_d);

    cudaDeviceSynchronize();   
 
    // Transfer results to host
    dtype rowSum, colSum;
    gpuErrchk(cudaMemcpy(&rowSum, dists_d, sizeof(dtype), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&colSum, dists2_d, sizeof(dtype), cudaMemcpyDeviceToHost));

    // Sum of means of distances
    dtype distance = rowSum/m + colSum/n;

    gpuErrchk(cudaFree(s1_d));
    gpuErrchk(cudaFree(s2_d));
    gpuErrchk(cudaFree(dists_d));
    gpuErrchk(cudaFree(dists2_d));
    
    if (verbose) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU computation time: %.3f ms\n", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return distance;
}


void readPointsFromCSV(FILE *f, dtype *points, int numPoints, int dims) {
    for (int i = 0; i < numPoints; i++) {
        fscanf(f, " %f", &points[i*dims]);
        for (int j = 1; j < dims; j++) {
            fscanf(f, ",%f", &points[i*dims + j]);
        }
    }
}

int countColumns(const char *line) {
    int count = 1;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') {
            count++;
        }
    }
    return count;
}

#define MAX_LINE_LENGTH 20000

int getCSVDimensions(const char *filename, int *height, int *width) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", filename);
        return 0;
    }

    char line[MAX_LINE_LENGTH];
    *height = 0;
    *width = 0;

    while (fgets(line, sizeof(line), file)) {
        if (*height == 0) {
            *width = countColumns(line);
        }
        (*height)++;
    }

    fclose(file);
    return 1;
}


// Driver function
int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc != 3) {
        printf("Usage: %s <path to pointset A> <path to pointset B>\n", argv[0]);
        return 1;
    }

    int m, n, dims1, dims2;    
    int num_iter = 10, verbose = 0;

    getCSVDimensions(argv[1], &m, &dims1);
    getCSVDimensions(argv[2], &n, &dims2);

    if (dims1 != dims2) {
        fprintf(stderr, "Error: Pointsets have different dimensions (%d and %d)\n", dims1, dims2);
        return 1;
    }
    
    int dims = dims1;
 
    if (m > MAX_POINTS || n > MAX_POINTS) {
        fprintf(stderr, "Error: Maximum number of points exceeded (limit: %d)\n", MAX_POINTS);
        return 1;
    }
    
    if (dims > MAX_DIMS) {
        fprintf(stderr, "Error: Maximum dimensions exceeded (limit: %d)\n", MAX_DIMS);
        return 1;
    }

    dtype *s1 = (dtype*)malloc(m * dims * sizeof(dtype));
    dtype *s2 = (dtype*)malloc(n * dims * sizeof(dtype));

    if (s1 == NULL || s2 == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(s1);
        free(s2);
        return 1;
    }

    FILE *f1 = fopen(argv[1], "r"),
         *f2 = fopen(argv[2], "r");
    readPointsFromCSV(f1, s1, m, dims);
    readPointsFromCSV(f2, s2, m, dims);
    fclose(f1);
    fclose(f2);

    // Warmup (avoids anomaly in first run)
    cudaDeviceSynchronize();

    // Benchmarking    
    double start_time = (double)clock()/CLOCKS_PER_SEC;
    dtype distance = computeChamferDistance(s1, m, s2, n, dims, verbose);
    double end_time = (double)clock()/CLOCKS_PER_SEC;

    printf("Chamfer Distance: %f\n", distance);
    printf("Time taken to compute Chamfer Distance: %f milliseconds\n", (end_time - start_time)*1000);

    free(s1);
    free(s2);

    return 0;
}
