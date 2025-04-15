If you're on Linux/macOS:
Terminal

cd /path/to/your/file
g++ a.cpp -fopenmp -o a.out
./a.out

sudo apt update
sudo apt install g++

window
a.exe

4
nvcc main.cu -o matrix_mul
./matrix_mul

nvcc --version

Note: The file should have a .cu extension (not .cpp) for CUDA. So rename it to a.cu if needed:
mv a.cpp a.cu
