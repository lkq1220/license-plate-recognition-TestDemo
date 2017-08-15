/********************************************************************
*名称:plateDetect.h
*作者:D
*时间:2016.01.22
*功能:车牌检测头文件
*********************************************************************/
#ifndef PLATEDETECT_H
#define PLATEDETECT_H

/********************************************************************
*头文件
*********************************************************************/
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <ml.h>

/********************************************************************
*宏定义
*********************************************************************/
#define PLATE_MORP_X 21   //X卷积核
#define PLATE_MORP_Y 3    //Y卷积核

#define FLOODFILLNUM 10   //填充次数
#define FLOODFILLDIF 75   //填充误差

#define PLATEWIDTH 103   //车牌调整宽度
#define PLATEHEIGHT 33   //车牌调整高度

#define PLATEASPECT 3.1428   //车牌宽高比 = 440/140 = 3.1428
#define PLATEERROR 0.4       //宽高误差率 = 40%

/********************************************************************
*命名空间
*********************************************************************/
using namespace std;
using namespace cv;

/********************************************************************
*类型定义
*********************************************************************/
typedef struct plate{
	Mat    img;   //车牌图片
	Rect   pos;   //车牌位置
	string num;   //车牌号码
}PLATE;

/********************************************************************
*函数原型
*********************************************************************/
vector<PLATE> plateDetect(Mat sourceImage);
bool verifyPlate(RotatedRect rect);

#endif