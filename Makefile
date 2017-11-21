#ifndef CPP 
#	CPP =  g++
#endif
CPP=g++

CPPFLAGS=-O3 -lm

LIBS =  -lOpenCL -fopenmp -L /opt/rocm/opencl/lib/x86_64
#WARN = -Wno

COMMON_DIR = C_common

test: test.cpp $(COMMON_DIR)/wtime.c $(COMMON_DIR)/device_info.c
	$(CPP) $^ $(CPPFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@

clean:
	rm -rf test
