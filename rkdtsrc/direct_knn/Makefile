EXEC=test_lowmem.exe direct_bench.exe lowmem_perf.exe test_distance.exe

#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX) $< -o $@  $(LDFLAGS) 

.PHONY: clean

clean: 
	@-$(RM) *.o *.exe *.out *.d


