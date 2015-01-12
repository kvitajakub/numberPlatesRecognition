
SRC=detection.cpp Features_ocr.cpp Features_char.cpp recognition.cpp main.cpp
TARGET=spz
ARGUMENTS=-d input -o output

CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LIBS) $^ -o $@

clean:
	rm -f *.o *~ $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGUMENTS)