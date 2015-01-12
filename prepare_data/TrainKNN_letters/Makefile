
SRC=Features_ocr.cpp TrainKNN_letters.cpp 
TARGET=train
ARGUMENTS=-d "train_data"

CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@
	
test: Features_ocr.cpp testKNN_letters.cpp
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@

clean:
	rm -f *.o *~ $(TARGET) test

run: $(TARGET)
	./$(TARGET) $(ARGUMENTS)