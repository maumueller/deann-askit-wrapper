EXEC=lsh_allnearest.exe lsh_qNEr.exe lsh_smallQ.exe direct_largeQ.exe
#include	$(KNN_DIR)/makefile.in.files/makefile.in
include	$(KNN_DIR)/makefile.inc

all:	$(EXEC)

%.exe: %.o
	$(CXX) $(CXX_FLAGS) $< -o $@  $(LDFLAGS) 

.PHONY:	clean

clean: 
	@-$(RM) *.o *.exe *.out *.d

