CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all : CharsTrain.cpp
	@g++ $^ $(CFLAGS) $(LIBS) -o CharsTrain
	
clean:
	@rm -rf ./CharsTrain ./tmp/ann.xml
