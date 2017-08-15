CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all : CharsSegment.cpp
	@g++ $^ $(CFLAGS) $(LIBS) -o CharsSegment
	
clean:
	@rm -rf ./CharsSegment ./tmp/*.jpg
