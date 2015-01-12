all: $(patsubst %.cpp, %, $(wildcard *.cpp))

%: %.cpp 
	g++ `pkg-config --libs --cflags opencv` $^ -o $@  

clean:
	rm -rf *o
