
all:
	nvcc -x cu -std=c++17 main.cu -lcuda -lcudart -I/opt/cuda/include -L/opt/cuda/lib64 -Wno-deprecated-gpu-targets -o main
