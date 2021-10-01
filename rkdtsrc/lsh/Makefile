EXEC=test_partitionedK.exe \
 test_fromFile.exe test_alltoall.exe test_distinct.exe test_replicated.exe

#include	$(KNN_DIR)/makefile.in.files/makefile.in
include	$(KNN_DIR)/makefile.inc

#CXX=kinst-ompp icpc
#LDFLAGS += -L/usr/local/lib -llikwid


all:	$(EXEC)

%.exe: %.o
	$(CXX) $(CXX_FLAGS) $< -o $@  $(LDFLAGS) 

.PHONY:	clean

clean: 
	@-$(RM) *.o *.exe *.out *.d

#CXX = icpc
#OPT = -openmp
#CFLAGS = $(OPT) -fPIC -O3
#LFLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lguide -lpthread
#NOWARN = -Wno-deprecated
#
#include $(KNN_DIR)/makefile.in.files/makefile.in
#INC = $(ALL_INCLUDE)
#
#all: $(EXEC)
#
#%.exe: %.o 
#	$(CXX) $< -o $@  $(LDFLAGS) 
#
#.PHONY: clean
#
#clean: 
#	@-$(RM) *.o *.exe *.out *.d
#serial:
#	$(CXX) $(NOWARN) -O3 -o lsh_serial.exe lsh_serial.cpp
#	$(CXX) $(NOWARN) -O3 -DBRUTEFORCE -o lsh_serial_correctness.exe lsh_serial.cpp
#parblas:
#	$(CXX) $(NOWARN) $(LDFLAGS) $(CXXFLAGS) $(INC) -o lsh_par_blas.exe lsh_par_blas.cpp
#	$(CXX) $(NOWARN) $(LFLAGS) $(CXXFLAGS) $(INC) -DBRUTEFORCE -o lsh_par_blas_check_correctness.exe lsh_par_blas.cpp
#
#parnoblas:
#	$(CXX) $(NOWARN) $(CFLAGS) -o lsh_par_no_blas.exe lsh_par_no_blas.cpp
#	$(CXX) $(NOWARN) $(CFLAGS) -DBRUTEFORCE -o lsh_par_no_blas_check_correctness.exe lsh_par_no_blas.cpp
#
