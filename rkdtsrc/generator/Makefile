EXEC=genDataSet.exe

#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX) $< -o $@  $(LDFLAGS) 

.PHONY: clean

clean: 
	@-$(RM) *.o *.exe *.out *.d


