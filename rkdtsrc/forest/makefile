EXEC=test.exe test_show_bin.exe

#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX)  $< -o $ $(LOC_OBJ) $@ $(LDFLAGS) 

#.PHONY: clean

clean: 
		@-$(RM) *.o ${EXEC} *.out *.d


