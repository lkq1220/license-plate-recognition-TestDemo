/********************************************************************
*名称:plateDetect.cpp
*作者:D
*时间:2016.01.22
*功能:车牌检测
*********************************************************************/

/********************************************************************
*头文件
*********************************************************************/
#include "plateDetect.h"

/********************************************************************
*名称:plateDetect
*参数:
*	sourceImage   源图片
*返回:
*	plates   车牌
*功能:车牌检测
*********************************************************************/
vector<PLATE> plateDetect(Mat sourceImage){
	//灰度变换
	Mat grayImage;
	
	cvtColor(sourceImage, grayImage, CV_BGR2GRAY);
	
	//均值滤波
	blur(grayImage, grayImage, Size(5, 5));
	
	//垂直检测
	Sobel(grayImage, grayImage, CV_8U, 1, 0, 3);
	
	//二值变换
	threshold(grayImage, grayImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	
	//形态变换
	Mat element;
	
	element = getStructuringElement(MORPH_RECT, Size(PLATE_MORP_X, PLATE_MORP_Y));
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, element);
	
	//获取轮廓
	vector<vector<Point> > contours;
	
	findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//轮廓裁剪
	vector<vector<Point> >::iterator contoursItr;
	vector<RotatedRect> rects;

	contoursItr = contours.begin();
	while( contoursItr != contours.end() ){
		RotatedRect minRect;

		minRect = minAreaRect(Mat(*contoursItr));
		if(verifyPlate(minRect)){
			rects.push_back(minRect);
		}

		contoursItr++;
	}
	
	vector<PLATE> plates;
	
	for(int i = 0; i < rects.size(); i++){
		//漫水填充
		float width, height, length;
		
		width = rects[i].size.width;
		height = rects[i].size.height;
		length = (height < width) ? height : width;
		length = length / 2;
		
		Mat mask;
		Point seed;
		Rect comp;
		Scalar newVal, loDiff, upDiff;
		int flags;
		
		mask = Mat::zeros(sourceImage.rows + 2, sourceImage.cols + 2, CV_8UC1);
		srand(time(0));
		newVal = Scalar(255, 255 ,255);
		loDiff = Scalar(FLOODFILLDIF, FLOODFILLDIF, FLOODFILLDIF);
		upDiff = Scalar(FLOODFILLDIF, FLOODFILLDIF, FLOODFILLDIF);
		flags = 4 | (255 << 8) | CV_FLOODFILL_FIXED_RANGE | CV_FLOODFILL_MASK_ONLY;
		
		for(int j = 0; j < FLOODFILLNUM; j++){
			//在原点附近取随机种子
			seed.x = (rects[i].center.x) + (rand()%(int)length) - (length/2);
			seed.y = (rects[i].center.y) + (rand()%(int)length) - (length/2);
			
			floodFill(sourceImage, mask, seed, newVal, &comp, loDiff, upDiff, flags);
		}
		
		//获取连通区
		Mat_<uchar>::iterator maskItr;
		vector<Point> pointInterest;

		maskItr = mask.begin<uchar>();
		while( maskItr != mask.end<uchar>() ){
			if(*maskItr == 255){
				pointInterest.push_back(maskItr.pos());
			}
			
			maskItr++;
		}
		
		//裁剪连通区
		RotatedRect minRect;

		minRect = minAreaRect(Mat(pointInterest));
		if(!verifyPlate(minRect)){
			continue;
		}
		
		//设置矩形
		Point2f rectCenter;
		Size2f rectSize;
		float rectAngle;
		
		rectCenter = minRect.center;
		rectSize = minRect.size;
		rectAngle = minRect.angle;
		
		if(minRect.size.width < minRect.size.height){   //如果矩形宽小于高，旋转90度
			swap(rectSize.width, rectSize.height);
			rectAngle = rectAngle + 90;
		}
		
		//旋转图片
		Mat rotateMat;
		Mat rotateImage;
		
		rotateMat = getRotationMatrix2D(rectCenter, rectAngle, 1);
		warpAffine(sourceImage, rotateImage, rotateMat, sourceImage.size(), INTER_CUBIC);
		
		//裁剪图片
		getRectSubPix(rotateImage, rectSize, rectCenter, rotateImage);
		
		//调整大小
		Mat resizeImage;
		
		resizeImage.create(PLATEHEIGHT, PLATEWIDTH, CV_8UC3);
		resize(rotateImage, resizeImage, resizeImage.size(), 0, 0, INTER_CUBIC);
		
		//归一图片
		Mat histImage;
		
		cvtColor(resizeImage, histImage, CV_BGR2GRAY);
		blur(histImage, histImage, Size(3, 3));
		equalizeHist(histImage, histImage);
		
 		//载入支持向量
		CvSVM svmClassifier;
	
		svmClassifier.load("svm.xml");
		
		//提取亮度特征
		Mat featureMat;
		
		featureMat = histImage.reshape(1, 1);
		featureMat.convertTo(featureMat, CV_32FC1);
		
		//车牌检测
		float classes;
		
		classes = svmClassifier.predict(featureMat);
		if(classes == 1){
			PLATE temp;
			
			temp.img = histImage;
			temp.pos = minRect.boundingRect();
			
			plates.push_back(temp);
		}
	}	
	
	return plates;
}

/********************************************************************
*名称:verifyPlate
*参数:
*	rect   矩形区域
*返回:
*	flag   1 满足
*	       0 不满足
*功能:检测区域是否满足车牌特征
*********************************************************************/
bool verifyPlate(RotatedRect rect){
	//设置矩形宽高
	float width, height;
	
	width = rect.size.width;
	height = rect.size.height;
	
	//宽高比例检测
	float platAspect, miniAspect, maxiAspect;
	
	if(width > height){
		platAspect = width / height;
	}else{
		platAspect = height / width;
	}
	miniAspect = PLATEASPECT - (PLATEASPECT * PLATEERROR);
	maxiAspect = PLATEASPECT + (PLATEASPECT * PLATEERROR);
	
	if(platAspect < miniAspect || maxiAspect < platAspect){
		return false;
	}
	
	//面积大小检测
	float platArea, miniArea, maxiArea;
	
	platArea = width * height;
	miniArea = (PLATEASPECT * 15) * 15;
	maxiArea = (PLATEASPECT * 125) * 125;
	
	if(platArea < miniArea || maxiArea < platArea){
		return false;
	}
	
	return true;
}