
SRC=Features_char.cpp TrainMSER.cpp
TARGET=train
ARGUMENTS=-l letters -n notletters

CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@
	
test: Features_char.cpp testSVM_mser.cpp
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@

clean:
	rm -f *.o *~ $(TARGET) test

run: $(TARGET)
	./$(TARGET) $(ARGUMENTS)