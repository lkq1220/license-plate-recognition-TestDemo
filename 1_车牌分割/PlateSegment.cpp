/********************************************************************
*名称:PlateSegment.c
*作者:D
*时间:2016.01.18
*功能:车牌分割
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
#define MORPHOLOGY_X 21   //X卷积核
#define MORPHOLOGY_Y 3    //Y卷积核

#define FLOODFILLNUM 10   //填充次数
#define FLOODFILLDIF 75   //填充误差

#define PLATEWIDTH 103   //车牌调整宽度
#define PLATEHEIGHT 33   //车牌调整高度

#define PLATEASPECT 3.1428   //车牌宽高比 = 440/140 = 3.1428
#define ASPECTERROR 0.4      //宽高误差率 = 40%


/********************************************************************
*命名空间
*********************************************************************/
using namespace std;
using namespace cv;

/********************************************************************
*函数原型
*********************************************************************/
int main(int argc, char **argv);
bool verifyPlate(RotatedRect rect);
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
		cout << "Usage:\n./PlateSegment <image>\n";
		return -1;
	}
	
	//载入图像
	Mat sourceImage;
	
	sourceImage = imread(argv[1]);
	
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
	
	element = getStructuringElement(MORPH_RECT, Size(MORPHOLOGY_X, MORPHOLOGY_Y));
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
		
		//保存图像
		string sourceFile;
		stringstream distanceFile;
		
		sourceFile = getFilename(argv[1]);
		distanceFile << "tmp/" << sourceFile << "_" << i << ".jpg";
		imwrite(distanceFile.str(), histImage);
	}

	return 0;
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
	miniAspect = PLATEASPECT - (PLATEASPECT * ASPECTERROR);
	maxiAspect = PLATEASPECT + (PLATEASPECT * ASPECTERROR);
	
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