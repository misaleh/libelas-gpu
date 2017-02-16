main_gpu.o: GPU/descriptor.cu CPU/matrix.cpp CPU/triangle.cpp  CPU/elas.cpp GPU/elas_gpu.cu CPU/elas.h
	nvcc -w -O3 -arch sm_20   `pkg-config --cflags --libs opencv` -lopencv_gpu GPU/descriptor.cu CPU/matrix.cpp CPU/triangle.cpp  CPU/elas.cpp GPU/elas_gpu.cu  main_gpu.cu   -o main_gpu.o
run:
	time ./main_gpu.o
clean:
	rm -f main_gpu.o
