#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

KNN_LIB_OBJ = $(KNN_SRC:.cpp=.o)

ARFILES = libknn.a
all: $(ARFILES)

libknn.a: $(KNN_LIB_OBJ) 
	$(AR) $(ARFLAGS) $(KNN_DIR)/build/libknn.a $(KNN_LIB_OBJ)
	$(RANLIB) $(KNN_DIR)/build/libknn.a

clean:
	-$(RM) *.a

