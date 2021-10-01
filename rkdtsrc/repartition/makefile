EXEC=test_treerepart.exe
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX) $< -o $@  $(LDFLAGS) 

.PHONY: clean

clean: 
	@-$(RM) *.o *.exe *.out *.d


