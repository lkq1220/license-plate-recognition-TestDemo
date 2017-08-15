CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all : PlateSegment.cpp
	@g++ $^ $(CFLAGS) $(LIBS) -o PlateSegment
	
clean:
	@rm -rf ./PlateSegment ./tmp/*.jpg
