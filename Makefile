NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_61

matmul: src/matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: run clean
run: matmul
	./matmul 512

clean:
	rm -f matmul
