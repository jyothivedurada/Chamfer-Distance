# Chamfer-Distance

To run the code, compile with `nvcc chamfer.cu -o ./chamfer.out`
For profiling use `ncu -f -k "regex:<kernel name>" -o profile --target-processes all --page details ./chamfer.out`

The python script `test.py` is for verifying the answer computed by `chamfer.cu`