CXX = g++
COMMON = `pkg-config --cflags opencv`
LIB = ../ncnn/lib/libncnn.a
LIB += `pkg-config --libs opencv`
INCLUDE = -I../ncnn/include
COMMON += -fopenmp -pthread
DEPS = $(wildcard *.h)
OBJ = base.o mtcnn.o arcface.o
all : main
main : main.cpp $(OBJ)
	$(CXX) $(COMMON) $(INCLUDE) $^ -o $@ $(LIB)
%.o : %.cpp $(DEPS)
	$(CXX) $(COMMON) $(INCLUDE) -c $< -o $@
.PHONY : clean
clean :
	rm -rf $(OBJ) main
