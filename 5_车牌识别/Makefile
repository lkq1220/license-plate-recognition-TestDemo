CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all : main.cpp plateDetect.cpp charsDetect.cpp
	@g++ $^ $(CFLAGS) $(LIBS) -o PlateRecognition
	
clean:
	@rm -rf ./PlateRecognition ./tmp/*.jpg
