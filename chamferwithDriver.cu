#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define dtype float
#define DTYPE_MAX FLT_MAX
#define FULL_MASK 0xffffffff
#define PROFILE

#define M 30
#define N 40
#define DIMS 128

#define TILE_SIZE 32
#define WORK_SIZE 10

#define MAX_POINTS 10000
#define MAX_DIMS 1024

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Computes the squared difference between components of vectors in s1 and s2.
// 
// s1: Array of `m` vectors, each with `dims` dimensions
// s2: Array of `n` vectors, each with `dims` dimensions
// res: Result: `m * n * dims` array of squared differences
__global__ void computeSqrDiffs(int m, int n, int dims, dtype *s1, dtype *s2, dtype *res) {
    extern __shared__ dtype s2_data[];
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int k_local = k%32;

    int group = blockIdx.z;
    int starting_j = group*WORK_SIZE;
    int stopping_j = (group+1)*WORK_SIZE; // Branchless minimum of n and (group+1) * WORK_SIZE
    int local_s2_offset = -starting_j * blockDim.x;
    stopping_j = n ^ ((stopping_j ^ n) & -(stopping_j < n));

    if (k >= dims) return;

    for (int j = starting_j + threadIdx.y; j < stopping_j; j += blockDim.y) { 
        s2_data[local_s2_offset + j*blockDim.x + k_local] = s2[j*dims + k];
    }

    if (i >= m) return;
    dtype s1_elem = s1[i*dims + k];

    __syncthreads();   

    for (int j = starting_j; j < stopping_j; j++) {
        dtype diff = s1_elem - s2_data[local_s2_offset + j*blockDim.x + k_local];
        res[(i * n * dims) + (j * dims) + k] = diff * diff;
    }
}

__global__ void sumDiffs(int m, int n, int dims, dtype *sqrDiffs, dtype *res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / 32;

    int i = wid % m;
    int j = wid / m;

    if (i >= m || j >= n) return;

    dtype sum = 0;
    for (int k = tid % 32; k < dims; k += 32) {
        sum += sqrDiffs[i*n*dims + j*dims + k];
    }
    
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4);
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    
    if (tid % 32 == 0) res[i*n + j] = sqrt(sum);
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
      
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    if (verbose) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    dtype *s1_d, *s2_d, *sqrDiffs_d, *dists_d, *dists2_d;

    gpuErrchk(cudaMalloc(&s1_d, m * dims * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&s2_d, n * dims * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&sqrDiffs_d, m * n * dims * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&dists_d, m * n * sizeof(dtype)));
    gpuErrchk(cudaMalloc(&dists2_d, m * n * sizeof(dtype)));

    gpuErrchk(cudaMemcpy(s1_d, s1_h, m * dims * sizeof(dtype), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(s2_d, s2_h, n * dims * sizeof(dtype), cudaMemcpyHostToDevice));

    // Compute squared differences
    computeSqrDiffs<<<dim3((dims+31)/32, (m+31)/32, (n+WORK_SIZE-1)/WORK_SIZE), dim3(32, 32, 1), sizeof(dtype) * 32 * WORK_SIZE>>>(
        m, n, dims, s1_d, s2_d, sqrDiffs_d);
 
    // Compute distances
    sumDiffs<<<(32 * m * n + 1023) / 1024, 1024>>>(m, n, dims, sqrDiffs_d, dists_d);   

    // Copy distances
    gpuErrchk(cudaMemcpy(dists2_d, dists_d, m * n * sizeof(dtype), cudaMemcpyDeviceToDevice));

    // Reductions along the two dimensions can be done parallely on different streams
    minReduceEachRow<<<m, (((n-1)/32)+1)*32, 0, stream1>>>(m, n, dims, dists_d);
    minReduceEachCol<<<n, (((m-1)/32)+1)*32, 0, stream2>>>(m, n, dims, dists2_d); 

    sumRowMins<<<1, m, 0, stream1>>>(m, n, dims, dists_d);
    sumColMins<<<1, n, 0, stream2>>>(m, n, dims, dists2_d);

    gpuErrchk(cudaStreamSynchronize(stream1));
    gpuErrchk(cudaStreamSynchronize(stream2));

    // Transfer results to host
    dtype rowSum, colSum;
    gpuErrchk(cudaMemcpy(&rowSum, dists_d, sizeof(dtype), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&colSum, dists2_d, sizeof(dtype), cudaMemcpyDeviceToHost));

    dtype distance = rowSum/m + colSum/n;

    gpuErrchk(cudaFree(s1_d));
    gpuErrchk(cudaFree(s2_d));
    gpuErrchk(cudaFree(sqrDiffs_d));
    gpuErrchk(cudaFree(dists_d));
    gpuErrchk(cudaFree(dists2_d));
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
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

// Generate random points within a specified range
void generateRandomPoints(dtype *points, int numPoints, int dims, float range) {
    for (int i = 0; i < numPoints * dims; i++) {
        points[i] = ((dtype)rand() / RAND_MAX) * range;
    }
}

// Print usage information
void printUsage(const char *programName) {
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -m <int>     : Number of points in first set (default: %d)\n", M);
    printf("  -n <int>     : Number of points in second set (default: %d)\n", N);
    printf("  -d <int>     : Dimensions of points (default: %d)\n", DIMS);
    printf("  -r <float>   : Range for random point values [0, range] (default: 1.0)\n");
    printf("  -i <int>     : Number of iterations (default: 10, 1 if profiling)\n");
    printf("  -v           : Verbose output\n");
    printf("  -vv          : Extra verbose output (prints distance matrix)\n");
    printf("  -h           : Show this help message\n");
}

// Driver function
int main(int argc, char *argv[]) {
    int m = M;
    int n = N;
    int dims = DIMS;
    float range = 1.0f;
    int verbose = 0;
    int num_iter = 10;

    #ifdef PROFILE
        num_iter = 1;
    #endif

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) {
            dims = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i+1 < argc) {
            range = atof(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) {
            num_iter = atoi(argv[++i]);
            #ifdef PROFILE
                printf("Note: In PROFILE mode, iterations are fixed to 1\n");
                num_iter = 1;
            #endif
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-vv") == 0) {
            verbose = 2;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (m <= 0 || n <= 0 || dims <= 0) {
        fprintf(stderr, "Error: Invalid point set parameters\n");
        return 1;
    }
    
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

    srand(time(NULL));

    generateRandomPoints(s1, m, dims, range);
    generateRandomPoints(s2, n, dims, range);

    /*
    FILE *f1 = fopen("public_small2_pointsetA.csv", "r"),
         *f2 = fopen("public_small2_pointsetB.csv", "r");
    readPointsFromCSV(f1, s1, m, dims);
    readPointsFromCSV(f2, s2, m, dims);
    fclose(f1);
    fclose(f2);
    */

    if (verbose) {
        printf("Points 1: %d points, %d dimensions\n", m, dims);
        printf("Points 2: %d points, %d dimensions\n", n, dims);
        
        if (verbose > 1) {
            printf("\nPoints 1 (first few):\n");
            for (int i = 0; i < min(5, m); i++) {
                printf("  ");
                for (int j = 0; j < min(5, dims); j++) {
                    printf("%f ", s1[i * dims + j]);
                }
                if (dims > 5) printf("...");
                printf("\n");
            }
            
            printf("\nPoints 2 (first few):\n");
            for (int i = 0; i < min(5, n); i++) {
                printf("  ");
                for (int j = 0; j < min(5, dims); j++) {
                    printf("%f ", s2[i * dims + j]);
                }
                if (dims > 5) printf("...");
                printf("\n");
            }
        }
    }

    printf("m = %d, n = %d, dims = %d, num_iter = %d\n", m, n, dims, num_iter);     
    double total_time = 0;

    for (int i = 0; i < num_iter; i++) {
        double start_time = (double)clock()/CLOCKS_PER_SEC;
        
        dtype distance = computeChamferDistance(s1, m, s2, n, dims, verbose);
        
        double end_time = (double)clock()/CLOCKS_PER_SEC;
        double iteration_time = end_time - start_time;
        total_time += iteration_time;
        
        printf("Iteration %d - Chamfer Distance: %f (Time: %.6f seconds)\n", 
               i+1, distance, iteration_time);
    }

    printf("\nSummary:\n");
    printf("%d iterations completed in %.6f seconds.\n", num_iter, total_time);
    printf("Average time per iteration: %.6f seconds.\n", total_time / num_iter);

    free(s1);
    free(s2);

    return 0;
}
