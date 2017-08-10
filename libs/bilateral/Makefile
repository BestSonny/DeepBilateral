all: src/bilateral_slice_launcher.cu.o
	python build.py

src/bilateral_slice_launcher.cu.o: src/bilateral_slice_launcher.cu
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean && rm -rf _ext
