#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define dtype float

#define TILE_SIZE 32
#define MAX_POINTS 10000
#define MAX_DIMS 100


__global__ void computeSqrDiffs(int m, int n, int dims, dtype *s1, dtype *s2, dtype *res) {
    __shared__ dtype v1s[32], v2s[32];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i >= m || j >= n) return;
   
    if (threadIdx.y == 0) {
        v1s[threadIdx.x] = s1[i*dims + k];
    }

    if (threadIdx.x == 0) {
        v2s[threadIdx.y] = s2[j*dims + k];
    }

    __syncthreads();

    dtype diff = v1s[threadIdx.x] - v2s[threadIdx.y];
    int x = (i * n * dims) + (j * dims) + k;
    res[x] = diff * diff;
}

__global__ void sumDiffs(int m, int n, int dims, dtype *sqrDiffs, dtype *res) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= m || j >= n) return;
    dtype *vec = &sqrDiffs[(i * n * dims) + (j * dims)];
 
    for (int stride = 1; stride < dims; stride *= 2) {
        if (k % (2 * stride) == 0 && k + stride < dims) {
            vec[k] += vec[k + stride];
        }
        __syncthreads();
    } 

    res[i*n + j] = vec[0];
}

__global__ void minReduceEachRow(int m, int n, int dims, dtype *dists) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int stride = 1; stride < n; stride *= 2) {
        if (j % (2 * stride) == 0 && j + stride < n) {
            dists[n*i + j] = min(dists[n*i + j], dists[n*i + j + stride]);
        }
        __syncthreads();
    } 
}

__global__ void minReduceEachCol(int m, int n, int dims, dtype *dists) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;

    for (int stride = 1; stride < m; stride *= 2) {
        if (i % (2 * stride) == 0 && i + stride < m) {
            dists[i*n + j] = min(dists[i*n + j], dists[(i + stride)*n + j]);
        }
        __syncthreads();
    } 
}

__global__ void sumRowMins(int m, int n, int dims, dtype *dists) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;

    for (int stride = 1; stride < m; stride *= 2) {
        if (i % (2 * stride) == 0 && i + stride < m) {
            dists[i*n + j] = dists[i*n + j] + dists[(i + stride)*n + j];
        }
        __syncthreads();
    }
}

__global__ void sumColMins(int m, int n, int dims, dtype *dists) { 
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int stride = 1; stride < n; stride *= 2) {
        if (j % (2 * stride) == 0 && j + stride < n) {
            dists[n*i + j] = dists[n*i + j] + dists[n*i + j + stride];
        }
        __syncthreads();
    } 
}

dtype computeChamferDistance(dtype *s1, int m, dtype *s2, int n, int dims, int verbose) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    if (verbose) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    dtype *s1_d, *s2_d, *sqrDiffs_d, *dists_d, *dists2_d;
    
    cudaMalloc((void**)&s1_d, m * dims * sizeof(dtype));
    cudaMalloc((void**)&s2_d, n * dims * sizeof(dtype));
    cudaMalloc((void**)&sqrDiffs_d, m * n * dims * sizeof(dtype));
    cudaMalloc((void**)&dists_d, m * n * sizeof(dtype));
    cudaMalloc((void**)&dists2_d, m * n * sizeof(dtype));
    
    cudaMemcpy(s1_d, s1, m * dims * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(s2_d, s2, n * dims * sizeof(dtype), cudaMemcpyHostToDevice);
    
    computeSqrDiffs<<<dim3((m + TILE_SIZE - 1) / TILE_SIZE, 
                          (n + TILE_SIZE - 1) / TILE_SIZE, 
                          dims), 
                     dim3(TILE_SIZE, TILE_SIZE, 1)>>>(
        m, n, dims, s1_d, s2_d, sqrDiffs_d);
    
    sumDiffs<<<dim3(m, n, 1), 1024>>>(m, n, dims, sqrDiffs_d, dists_d);
    
    if (verbose > 1) {
        dtype *dists_h = (dtype*)malloc(m * n * sizeof(dtype));
        cudaMemcpy(dists_h, dists_d, m * n * sizeof(dtype), cudaMemcpyDeviceToHost);
        printf("Distances matrix:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", dists_h[i*n + j]);
            }
            printf("\n");
        }
        free(dists_h);
    }
    
    cudaMemcpy(dists2_d, dists_d, m * n * sizeof(dtype), cudaMemcpyDeviceToDevice);
    
    minReduceEachRow<<<dim3(m, 1, 1), dim3(1, n, 1), 0, stream1>>>(m, n, dims, dists_d);
    sumRowMins<<<dim3(1, n, 1), dim3(m, 1, 1), 0, stream1>>>(m, n, dims, dists_d);
    
    minReduceEachCol<<<dim3(1, n, 1), dim3(m, 1, 1), 0, stream2>>>(m, n, dims, dists2_d);
    sumColMins<<<dim3(m, 1, 1), dim3(1, n, 1), 0, stream2>>>(m, n, dims, dists2_d);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    dtype rowSum, colSum;
    cudaMemcpy(&rowSum, dists_d, sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy(&colSum, dists2_d, sizeof(dtype), cudaMemcpyDeviceToHost);
    
    dtype distance = rowSum + colSum;
    
    cudaFree(s1_d);
    cudaFree(s2_d);
    cudaFree(sqrDiffs_d);
    cudaFree(dists_d);
    cudaFree(dists2_d);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    if (verbose) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Computation time: %.3f ms\n", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return distance;
}

void generateRandomPoints(dtype *points, int numPoints, int dims, float range) {
    for (int i = 0; i < numPoints * dims; i++) {
        points[i] = ((dtype)rand() / RAND_MAX) * range;
    }
}

void printUsage(const char *programName) {
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -m <int>     : Number of points in first set\n");
    printf("  -n <int>     : Number of points in second set\n");
    printf("  -d <int>     : Dimensions of points\n");
    printf("  -r <float>   : Range for random point values [0, range]\n");
    printf("  -v           : Verbose output\n");
    printf("  -vv          : Extra verbose output (prints distance matrix)\n");
    printf("  -h           : Show this help message\n");
}

int main(int argc, char *argv[]) {
    int m = 100; 
    int n = 100; 
    int dims = 3;       
    float range = 10.0f;  
    int verbose = 0;      

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) {
            dims = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i+1 < argc) {
            range = atof(argv[++i]);
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
    
    dtype *points1 = (dtype*)malloc(m * dims * sizeof(dtype));
    dtype *points2 = (dtype*)malloc(n * dims * sizeof(dtype));
    
    srand(time(NULL));
    
    generateRandomPoints(points1, m, dims, range);
    generateRandomPoints(points2, n, dims, range);
    
    if (verbose) {
        printf("Points 1: %d points, %d dimensions\n", m, dims);
        printf("Points 2: %d points, %d dimensions\n", n, dims);
        
        if (verbose > 1 && m <= 10 && n <= 10) {
            printf("\nPoints 1:\n");
            for (int i = 0; i < m; i++) {
                printf("  ");
                for (int j = 0; j < dims; j++) {
                    printf("%f ", points1[i * dims + j]);
                }
                printf("\n");
            }
            
            printf("\nPoints 2:\n");
            for (int i = 0; i < n; i++) {
                printf("  ");
                for (int j = 0; j < dims; j++) {
                    printf("%f ", points2[i * dims + j]);
                }
                printf("\n");
            }
        }
    }
    
    dtype chamferDistance = computeChamferDistance(points1, m, points2, n, dims, verbose);
    
    printf("Chamfer Distance: %f\n", chamferDistance);
    
    free(points1);
    free(points2);
    
    return 0;
}