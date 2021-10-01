
Run := test.exe
Objs := fks_ompNode.o fksTree.o test.o

CXX := icpc
CC := icc
MPICXX := mpic++
CPPFLAGS +=  -Wall -DTIXML_USE_STL
CFLAGS += -openmp -fPIC -O3

KNN_DIR := /h2/bo/git.ronaldo/knn/src
#KNN_DIR := /h1/march/knn/src
KNN_INCLUDE := -I${KNN_DIR}/generator/ \
			   -I$(KNN_DIR)/include/ \
	           -I$(KNN_DIR)/include/binTree/ \
	           -I$(KNN_DIR)/include/direct_knn/ \
	           -I$(KNN_DIR)/include/repartition/ \
	           -I$(KNN_DIR)/include/lsh/ \
	           -I$(KNN_DIR)/include/parallelIO/ \
			   -I${KNN_DIR}/include/stTree/
KNN_LIB := -L$(KNN_DIR)/build -lknn -lrrot

CMD_INCLUDE := -I${KNN_DIR}/external/cmd/
CMD_LIB := -L/${KNN_DIR}/build -lcmd

BLAS_INCLUDE := -I${MKLROOT}/include
BLAS_LIBS := -L${MKLROOT}/lib/intel64 -mkl=parallel
#BLAS_INCLUDE := -I${ICES_MKL_INC}/
#BLAS_LIBS := -L${ICES_MKL_LIB}/ -mkl=parallel

ALL_INCLUDE += -I./ $(CMD_INCLUDE) $(KNN_INCLUDE) ${BLAS_INCLUDE}
LDFLAGS += -L./ -lpthread $(CMD_LIB) $(KNN_LIB) $(BLAS_LIBS)


.PHONY: all
all: clean $(Run)

.PHONY: clean
$(Run): $(Objs)
	$(MPICXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(Objs) $(LDFLAGS)
#	mv $(Run) ../bin/	
$(Objs): %.o: %.cpp
	$(MPICXX) -c $(CPPFLAGS) $(CFLAGS) $< -o $@ $(ALL_INCLUDE)

clean:
	rm -rf $(Run) $(Objs)

