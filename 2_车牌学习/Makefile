CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all : PlateTrain.cpp
	@g++ $^ $(CFLAGS) $(LIBS) -o PlateTrain
	
clean:
	@rm -rf ./PlateTrain ./tmp/*.xml
