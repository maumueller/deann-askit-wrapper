EXEC=test.exe 
#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)

#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX)  $< -o $ $(LOC_OBJ) $@ $(LDFLAGS) 

#.PHONY: clean

clean: 
		@-$(RM) *.o ${EXEC} *.out *.d


