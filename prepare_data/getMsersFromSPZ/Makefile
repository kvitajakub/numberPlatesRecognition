
SRC=Features_char.cpp get_msers.cpp
TARGET=getmsers
ARGUMENTS=-d input"

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