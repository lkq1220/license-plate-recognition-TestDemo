/********************************************************************
*名称:main.cpp
*作者:D
*时间:2016.01.22
*功能:车牌识别
*********************************************************************/

/********************************************************************
*头文件
*********************************************************************/
#include "plateDetect.h"
#include "charsDetect.h"

/********************************************************************
*名称:main
*参数:
*	argc   索引
*	argv   参数
*返回:
*	state    0 成功
*	        -1 失败
*功能:主函数
*********************************************************************/
int main(int argc, char **argv){
	//参数检查
	if(argc != 2){
		cout << "Usage:\n./PlateRecognition <image>\n";
		return -1;
	}
	
	//读取图片
	Mat sourceImage;
	
	sourceImage = imread(argv[1]);
	
	//车牌检测
	vector<PLATE> plates;
	
	plates = plateDetect(sourceImage);
	
	//字符识别
	for(int i = 0; i < plates.size(); i++){
		plates[i].num = charsDetect(plates[i].img);
	}
	
	//显示结果
	for(int i = 0; i < plates.size(); i++){
		Rect platePos;
		string plateNum;
		Point plateOrg;
		
		platePos = plates[i].pos;
		plateNum = plates[i].num;
		plateOrg = Point(plates[i].pos.x, plates[i].pos.y);
		
		rectangle(sourceImage, platePos, Scalar(0, 0, 255));
		putText(sourceImage, plateNum, plateOrg, CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	}
	namedWindow("PlateRecognition", CV_WINDOW_AUTOSIZE);
	imshow("PlateRecognition", sourceImage);
	
	waitKey(0);
	
	return 0;
}