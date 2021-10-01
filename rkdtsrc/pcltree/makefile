EXEC=test_pcltree.exe #test_hs.exe test_queryK.exe test_approxK.exe test_queryKGreedy.exe
#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)

include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX)  $< -o $ $(LOC_OBJ) $@ $(LDFLAGS) 

#.PHONY: clean

clean: 
		@-$(RM) *.o ${EXEC} *.out *.d


