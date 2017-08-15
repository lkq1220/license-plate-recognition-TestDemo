/********************************************************************
*名称:charsDetect.cpp
*作者:D
*时间:2016.01.25
*功能:字符识别
*********************************************************************/

/********************************************************************
*头文件
*********************************************************************/
#include "charsDetect.h"

/********************************************************************
*名称:charsDetect
*参数:
*	sourceImage   源图片
*返回:
*	plateNumber   车牌号
*功能:车牌检测
*********************************************************************/
string charsDetect(Mat sourceImage){
	//二值变化
	Mat thresholdImage;
	
	threshold(sourceImage, thresholdImage, 200, 255, CV_THRESH_BINARY);   //字符颜色变为白色，用于轮廓检测
	
	//获取轮廓
	Mat contoursImage;
	vector<vector<Point> > contours;
	
	thresholdImage.copyTo(contoursImage);
	findContours(contoursImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	//轮廓裁剪
	vector<vector<Point> >::iterator contoursItr;
	vector<CHARS> chars;
	
	contoursItr = contours.begin();
	while( contoursItr != contours.end() ){
		Rect minRect;

		minRect = boundingRect(Mat(*contoursItr));
		Mat rect(thresholdImage, minRect);
		if(verifyChar(rect)){
			//位置调整
			float width, height, length;
			Mat transformMat, warpImage;
			
			width = rect.cols;
			height = rect.rows;
			length = (height > width) ? height : width;
			
			transformMat = Mat::eye(2, 3, CV_32FC1);
			transformMat.at<float>(0, 2) = (length - width)/2;   //调整到图像中心
			transformMat.at<float>(1, 2) = (length - height)/2;
			
			warpImage.create(length, length, rect.type());
			warpAffine(rect, warpImage, transformMat, warpImage.size());
			
			//调整大小
			Mat resizeImage;
			
			resizeImage.create(CHARLENGTH, CHARLENGTH, warpImage.type());
			resize(warpImage, resizeImage, resizeImage.size());
			
 			//载入神经网络
			CvANN_MLP annClassifier;
		
			annClassifier.load("ann.xml");
			
			//提取特征向量
			Mat featureMat;
			
			featureMat = getFeatures(resizeImage);
			
			//字符识别
			Mat classesMat;

			classesMat.create(1, CHARNUMBER, CV_32FC1);
			annClassifier.predict(featureMat, classesMat);
			
			//字符转换
			Point maxLoc;
			
			minMaxLoc(classesMat, 0, 0, 0, &maxLoc);
			
			//字符保存
			CHARS temp;
			
			temp.cha = characters[maxLoc.x];
			temp.pos = minRect.x;
			
			chars.push_back(temp);
		}

		contoursItr++;
	}
	
	//插值排序
	for(int i = 0; i < chars.size() - 1; i++){
		for(int j = i + 1; j < chars.size(); j++){
			if(chars[j].pos < chars[i].pos){
				CHARS temp;
				
				temp = chars[i];
				chars[i] = chars[j];
				chars[j] = temp;
			}
		}
	}
	
	//字符连接
	string plateNumber;
	
 	for(int i = 1; i < chars.size(); i++){   //因为无法显示汉字，所以去掉第一个字符
		plateNumber += chars[i].cha;
	}
	
	return plateNumber;
}

/********************************************************************
*名称:verifyChar
*参数:
*	rect   矩形区域
*返回:
*	flag   1 满足
*	       0 不满足
*功能:检测区域是否满足字符特征
*********************************************************************/
bool verifyChar(Mat rect){
	//设置矩形宽高
	float width, height;

	width = rect.cols;
	height = rect.rows;
	
	//宽高比例检测
	float charAspect, miniAspect, maxiAspect;
	
	charAspect = width / height;
	miniAspect = MINIASPECT;
	maxiAspect = CHARASPECT + (CHARASPECT * CHARERROR);
	
	if(charAspect < miniAspect || maxiAspect < charAspect){
		return false;
	}
	
	//高度大小检测
	if(height < MINCHARHEIGHT || MXNCHARHEIGHT < height){
		return false;
	}
	
	//面积比例检测
	float charArea, rectArea, charPercent;
	
	charArea = countNonZero(rect);
	rectArea = width * height;
	
	charPercent = charArea / rectArea;
	
	if(charPercent < MINAREAPERCENT || MXNAREAPERCENT < charPercent){
		return false;
	}
	
	return true;
}

/********************************************************************
*名称:getFeatures
*参数:
*	charImage   字符图片
*返回:
*	featureMat   特征矩阵
*功能:提取字符特征
*********************************************************************/
Mat getFeatures(Mat charImage){
	//投影直方图
	Mat vhist, hhist;
	
	vhist = Mat::zeros(1, charImage.cols, CV_32FC1);
	for(int i = 0; i < charImage.cols; i++){
		vhist.at<float>(i) = countNonZero(charImage.col(i));
	}
	
	hhist = Mat::zeros(1, charImage.rows, CV_32FC1);
	for(int i = 0; i < charImage.rows; i++){
		hhist.at<float>(i) = countNonZero(charImage.row(i));
	}
	
	//归一直方图
	double maxVal;
	
	minMaxLoc(vhist, 0, &maxVal, 0, 0);
	if(maxVal > 0){
		vhist.convertTo(vhist, -1, 1.0/maxVal, 0);
	}
	
	minMaxLoc(hhist, 0, &maxVal, 0, 0);
	if(maxVal > 0){
		hhist.convertTo(hhist, -1, 1.0/maxVal, 0);
	}
	
	//低分辨率特征
	Mat resizeImage;
	
	resizeImage.create(LOWSLENGTH, LOWSLENGTH, charImage.type());
	resize(charImage, resizeImage, resizeImage.size());
	
	//设置特征矩阵
	int featureNum;
	Mat featureMat;
	int featureCol;
	
	featureNum = vhist.cols + hhist.cols + (resizeImage.rows * resizeImage.cols);
	featureMat = Mat::zeros(1, featureNum, CV_32FC1);
	featureCol = 0;
	
	for(int i = 0; i < vhist.cols; i++){
		featureMat.at<float>(featureCol) = vhist.at<float>(i);
		featureCol++;
	}
	for(int i = 0; i < hhist.cols; i++){
		featureMat.at<float>(featureCol) = hhist.at<float>(i);
		featureCol++;
	}
	for(int i = 0; i < resizeImage.rows; i++){
		for(int j = 0; j < resizeImage.cols; j++){
			featureMat.at<float>(featureCol) = (float)resizeImage.at<unsigned char>(i, j);
			featureCol++;
		}
	}
	
	return featureMat;
}