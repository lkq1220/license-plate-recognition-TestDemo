/********************************************************************
*名称:CharsSegment.c
*作者:D
*时间:2016.01.20
*功能:字符分割
*********************************************************************/

/********************************************************************
*头文件
*********************************************************************/
#include <iostream>
#include <cv.h>
#include <highgui.h>

/********************************************************************
*宏定义
*********************************************************************/
#define CHARLENGTH 20   //字符调整长度

#define MINIASPECT 0.1     //最小宽高比 = 0.1，字符'1'的宽高比约等于0.1
#define CHARASPECT 0.5     //字符宽高比 = 45/90 = 0.5
#define ASPECTERROR 0.35   //宽高误差率 = 35%

#define MINCHARHEIGHT 15   //最小字符高度 = 15 pixels
#define MXNCHARHEIGHT 28   //最大字符高度 = 28 pixels

#define MINAREAPERCENT 0.2   //最小面积百分比 = 20%
#define MXNAREAPERCENT 0.8   //最大面积百分比 = 80%

/********************************************************************
*命名空间
*********************************************************************/
using namespace std;
using namespace cv;

/********************************************************************
*函数原型
*********************************************************************/
int main(int argc, char **argv);
bool verifyChar(Mat rect);
string getFilename(string str);

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
		cout << "Usage:\n./CharsSegment <image>\n";
		return -1;
	}
	
	//载入图像
	Mat grayImage;
	
	grayImage = imread(argv[1], 0);

	//二值变化
	Mat thresholdImage;
	
	threshold(grayImage, thresholdImage, 200, 255, CV_THRESH_BINARY);   //字符颜色变为白色，用于轮廓检测
	
	//获取轮廓
	Mat contoursImage;
	vector<vector<Point> > contours;
	
	thresholdImage.copyTo(contoursImage);
	findContours(contoursImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	//轮廓裁剪
	vector<vector<Point> >::iterator contoursItr;
	int i = 0;
	
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
			
			//保存图像
			string sourceFile;
			stringstream distanceFile;
			
			sourceFile = getFilename(argv[1]);
			distanceFile << "tmp/" << sourceFile << "_" << i++ << ".jpg";
			imwrite(distanceFile.str(), resizeImage);
		}
		
		contoursItr++;
	}
	
	return 0;
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
	maxiAspect = CHARASPECT + (CHARASPECT * ASPECTERROR);
	
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
*名称:getFilename
*参数:
*	str   字符串
*返回:
*	filename   文件名
*功能:获取文件名
*********************************************************************/
string getFilename(string str){

    char sep = '/';
    char sepExt='.';

    size_t i = str.rfind(sep, str.length( ));
    if(i != string::npos){
        string filename= (str.substr(i+1, str.length( ) - i));
        size_t j = filename.rfind(sepExt, filename.length( ));
        if(i != string::npos){
            return filename.substr(0,j);
        }else{
            return filename;
        }
    }else{
        return "";
    }
}