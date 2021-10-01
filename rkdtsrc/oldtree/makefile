#EXEC=test_debug.exe 
EXEC=test_oldtree.exe test_ot_mtree.exe


#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)


#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)

include $(KNN_DIR)/makefile.in.files/makefile.in

all: $(EXEC)

%.exe: %.o 
	$(CXX)  $< -o $ $(LOC_OBJ) $@ $(LDFLAGS) 

#.PHONY: clean

clean: 
		@-$(RM) *.o ${EXEC} *.out *.d


