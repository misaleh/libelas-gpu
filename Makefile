#elascpu.o: 
#	g++ -o elascpu filter.cpp descriptor.cpp matrix.cpp triangle.cpp elas.cpp  
#elas_gpu.o: elascpu.o
#	nvcc -arch sm_20  main_gpu.o -o elas_gpu.o
main_gpu.o: 
	nvcc -w -O3 -arch sm_20   `pkg-config --cflags --libs opencv` -lopencv_gpu CPU/descriptor.cpp CPU/matrix.cpp CPU/triangle.cpp CPU/filter.cpp CPU/elas.cpp GPU/elas_gpu.cu  main_gpu.cu   -o main_gpu.o
run:
	time ./main_gpu.o
clean:
	rm -f main_gpu.o