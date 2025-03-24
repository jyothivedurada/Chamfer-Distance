#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

#define dtype float
#define DTYPE_MAX FLT_MAX
#define FULL_MASK 0xffffffff
#define PROFILE

#define M 30
#define N 40
#define DIMS 128

#define TILE_SIZE 32
#define WORK_SIZE 20

#define SUM_X 1
#define SUM_Y 1

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

    int group = blockIdx.z;
    int starting_j = group*WORK_SIZE;
    int stopping_j = (group+1)*WORK_SIZE; // Branchless minimum of n and (group+1) * WORK_SIZE
    int local_s2_offset = -starting_j * blockDim.x;
    stopping_j = n ^ ((stopping_j ^ n) & -(stopping_j < n));

    if (k >= dims) return;
    
    for (int j = starting_j + threadIdx.y; j < stopping_j; j += blockDim.y) {
        s2_data[local_s2_offset + j*blockDim.x + k] = s2[j*dims + k];
    }

    if (i >= m) return;
    dtype s1_elem = s1[i*dims + k];

    __syncthreads();   

    for (int j = starting_j; j < stopping_j; j++) {
        dtype diff = s1_elem - s2_data[local_s2_offset + j*blockDim.x + k];
        res[(i * n * dims) + (j * dims) + k] = diff * diff;
    }
}

__global__ void sumDiffs(int m, int n, int dims, dtype *sqrDiffs, dtype *res) {
    __shared__ dtype sharedMemory[32*SUM_X*SUM_Y];
    dtype* vec = sharedMemory;

    int i = blockIdx.y*SUM_X;
    int j_0 = blockIdx.z*SUM_Y;
    int j;
    int s_i = i+SUM_X;
    int s_j = j_0+SUM_Y;
    s_i = m ^ ((s_i ^ m) & -(s_i < m));
    s_j = n ^ ((s_j ^ n) & -(s_j < n));

    int k = threadIdx.x;
    int w = k >> 5;

    dtype v = 0;

    if (k >= dims) return;
    if (k <= 31) vec[k] = 0;

    for (; i < s_i; i++) {
        for (j=j_0; j < s_j; j++) {

            v = sqrDiffs[(i * n * dims) + (j * dims) + k];
            v += __shfl_down_sync(FULL_MASK, v, 16);
            v += __shfl_down_sync(FULL_MASK, v, 8);
            v += __shfl_down_sync(FULL_MASK, v, 4);
            v += __shfl_down_sync(FULL_MASK, v, 2);
            v += __shfl_down_sync(FULL_MASK, v, 1);
            if (k%32 == 0) vec[w] = v;

            vec += 32;
        }
    }

    k = k % 32;
    __syncthreads();

    for (int id = w; id < SUM_X*SUM_Y; id += blockDim.x>>5) {
        i = blockIdx.y + (id / SUM_Y);
        j = blockIdx.z + (id % SUM_Y);  
        vec = sharedMemory + 32*id;  
                
        v = vec[k];
        v += __shfl_down_sync(FULL_MASK, v, 16);
        v += __shfl_down_sync(FULL_MASK, v, 8);
        v += __shfl_down_sync(FULL_MASK, v, 4);
        v += __shfl_down_sync(FULL_MASK, v, 2);
        v += __shfl_down_sync(FULL_MASK, v, 1);

        if (k == 0) res[i*n + j] = v; 
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
    vec[i>>5] = v;

    __syncthreads();
    
    
    if (i > 31) return;

    v = (i > n>>5) * DTYPE_MAX + (i <= n>>5) * vec[i];
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1));
    dists[row*n] = v;
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
    vec[i>>5] = v;

    __syncthreads();
    if (i > 31) return;

    v = (i > m>>5) * DTYPE_MAX + (i <= m>>5) * vec[i];
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fminf(v, __shfl_down_sync(FULL_MASK, v, 1));
    dists[col] = v;
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
    vec[i>>5] = v;
    
    __syncthreads();
    if (i > 31) return;

    v = (i <= m>>5) * vec[i];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    dists[0] = v;
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
    vec[i>>5] = v;
    
    __syncthreads();
    if (i > 31) return;

    v = (i <= n>>5) * vec[i];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    dists[0] = v;
}

// TODO: Do proper profiling
dtype chamferDistance(int m, int n, int dims, dtype *s1_h, dtype *s2_h) {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
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
 
    /* 
    // DEBUG: Print squared differences
    dtype *sqrDiffs_h = (dtype*)malloc(m * n * dims * sizeof(dtype));
    cudaMemcpy(sqrDiffs_h, sqrDiffs_d, m * n * dims * sizeof(dtype), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m * n * dims; i++) {
        printf("%f ", sqrDiffs_h[i]);
    }
    printf("\n");
    
    free(sqrDiffs_h);
    */

    // Compute distances
    sumDiffs<<<dim3(1, m/SUM_X, n/SUM_Y), dim3(dims, 1, 1)>>>(m, n, dims, sqrDiffs_d, dists_d);

    /*
    // DEBUG: Print distances
    dtype *dists_h = (dtype*)malloc(m * n * sizeof(dtype));
    cudaMemcpy(dists_h, dists_d, m * n * sizeof(dtype), cudaMemcpyDeviceToHost);
    printf("Distances: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", dists_h[i*n + j]);
        }
        printf("\n");
    }
    free(dists_h);
    */

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

    dtype distance = rowSum + colSum;

    gpuErrchk(cudaFree(s1_d));
    gpuErrchk(cudaFree(s2_d));
    gpuErrchk(cudaFree(sqrDiffs_d));
    gpuErrchk(cudaFree(dists_d));
    gpuErrchk(cudaFree(dists2_d));

    return distance;
}

// Driver function
int main(void) {
    srand(time(NULL));
    int m = M, n = N, dims = DIMS;
    int num_iter = 10; 

    #ifdef PROFILE
        num_iter = 1;
    #endif

    dtype *s1 = (dtype*)malloc(m * dims * sizeof(dtype));
    dtype *s2 = (dtype*)malloc(n * dims * sizeof(dtype));

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < dims; k++) {
            s1[i*dims + k] = (dtype)rand()/RAND_MAX;
            // DEBUG: For Verification
            // printf("s1 %d %d %f\n", i, k, s1[i*dims + k]);
        }
    }
  
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < dims; k++) {
            s2[i*dims + k] = (dtype)rand()/RAND_MAX;
            // DEBUG: For Verification
            // printf("s2 %d %d %f\n", i, k, s2[i*dims + k]);
        }
    }

    printf("m = %d, n = %d, dims = %d, num_iter = %d\n", m, n, dims, num_iter);     
    double total_time = 0;

    for (int i = 0; i < num_iter; i++) {
        double start_time = (double)clock()/CLOCKS_PER_SEC;
        
        dtype distance = chamferDistance(m, n, dims, (dtype*)s1, (dtype*)s2);
        printf("%f\n", distance);       
 
        double end_time = (double)clock()/CLOCKS_PER_SEC;
        total_time += end_time - start_time;
        
    }

    printf("%d iterations in %lf seconds.\n", num_iter, total_time);
    printf("%lf seconds per iteration.\n", total_time / num_iter);

    free(s1);
    free(s2);

    return;
}
