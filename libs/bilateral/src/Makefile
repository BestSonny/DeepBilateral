CONFIG_FILE := ../Makefile.config
include $(CONFIG_FILE)

all: bilateral_slice_launcher.cu bilateral_slice_launcher.h
	nvcc -c -o bilateral_slice_launcher.cu.o bilateral_slice_launcher.cu -x cu -Xcompiler -fPIC -arch=sm_52 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
clean:
	rm -rf bilateral_slice_launcher.cu.o
