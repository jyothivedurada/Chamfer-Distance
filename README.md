# Chamfer-Distance

To run the code, compile with `nvcc chamfer.cu -o ./chamfer.out`
For profiling use `ncu -f -k "regex:<kernel name>" -o profile --target-processes all --page details ./chamfer.out`

The python script `test.py` is for verifying the answer computed by `chamfer.cu`


# To run Code with driver function


### Compilation

Compile the program with the NVIDIA CUDA compiler:

```bash
nvcc -O3 chamfer.cu -o chamfer
```

### Basic Usage

Run the program with default settings (100 points in each set, 3 dimensions):

```bash
./chamfer
```

### Command-line Options

- `-m <int>`: Number of points in the first set (default: 100)
- `-n <int>`: Number of points in the second set (default: 100)
- `-d <int>`: Dimensions of points (default: 3)
- `-r <float>`: Range for random point values `[0-range]` (default: 10.0)
- `-v`: Verbose output (timing information)
- `-vv`: Extra verbose output (prints distance matrix)
- `-h`: Show help message

### Examples

Run with 500 points in each set with 3 dimensions:

```bash
./chamfer -m 500 -n 500
```

Run with different point set sizes and 4 dimensions:

```bash
./chamfer -m 200 -n 300 -d 4
```

Use a larger range for random values and show timing information:

```bash
./chamfer -r 100.0 -v
```

Show the full distance matrix (only recommended for small sets):

```bash
./chamfer -m 10 -n 10 -vv
```