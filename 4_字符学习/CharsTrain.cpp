/********************************************************************
*名称:CharsTrain.c
*作者:D
*时间:2016.01.21
*功能:字符学习
*********************************************************************/

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
#define CHARNUMBER 35   //字符数目
#define HIDENUMBER 20   //隐藏结点数目
#define LOWSLENGTH 20   //低分辨率像素

#define TRAINSAMPLE 0.8   //训练样本比例

/********************************************************************
*命名空间
*********************************************************************/
using namespace std;
using namespace cv;

/********************************************************************
*全局变量
*********************************************************************/
const char characters[CHARNUMBER] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_'};

/********************************************************************
*函数原型
*********************************************************************/
int main(int argc, char **argv);
Mat getFeatures(Mat charImage);

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
	if(argc != (CHARNUMBER + 1)){
		cout << "Usage:\n./CharsTrain <Char Number> ...\n";
		return -1;
	}
 	
	//读取数据
	int charNumber[CHARNUMBER];
	string charPath;
	
	Mat images;
	vector<float> labels;
	
	charPath = "img/";
	for(int i = 0; i < CHARNUMBER; i++){
		charNumber[i] = atoi(argv[i+1]);
		
		for(int j = 0; j < (int)(charNumber[i] * TRAINSAMPLE); j++){
			stringstream charFile;
			Mat charImage, featureMat;
			
			charFile << charPath << characters[i] << "/" << j << ".jpg";
			charImage = imread(charFile.str(), 0);
			featureMat = getFeatures(charImage);
			
			images.push_back(featureMat);
			labels.push_back(i);
		}
	}
	
	//设置数据
	Mat trainingData;
	Mat trainingLabs;
	
	Mat(images).copyTo(trainingData);
	Mat(labels).copyTo(trainingLabs);
	
	trainingData.convertTo(trainingData, CV_32FC1);
	trainingLabs.convertTo(trainingLabs, CV_32FC1);
	
	//设置参数
	Mat layerSizes;
	Mat trainClasses;
	Mat sampleWeights;
	
	layerSizes.create(1, 3, CV_32SC1);
	layerSizes.at<int>(0)= trainingData.cols;   //特征结点
    layerSizes.at<int>(1)= HIDENUMBER;          //隐藏结点
    layerSizes.at<int>(2)= CHARNUMBER;          //类别结点
	
	trainClasses.create(trainingData.rows, CHARNUMBER, CV_32FC1);
	for(int i = 0; i < trainClasses.rows; i++){
		for(int j = 0; j < trainClasses.cols; j++){
			if(trainingLabs.at<float>(i) == j){
				trainClasses.at<float>(i, j) = 1;
			}else{
				trainClasses.at<float>(i, j) = 0;
			}
		}
	}
	
	sampleWeights = Mat::ones(1, trainingData.rows, CV_32FC1);
	
	//学习模型
	CvANN_MLP annClassifier;
	
	annClassifier.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
	annClassifier.train(trainingData, trainClasses, sampleWeights);
	
	//保存模型
	annClassifier.save("tmp/ann.xml");
	
	//测试模型
	int charsSample[CHARNUMBER];
	
	for(int i = 0; i < CHARNUMBER; i++){
		charsSample[i] = 0;
		
		for(int j = (int)(charNumber[i] * TRAINSAMPLE); j < charNumber[i]; j++){
			stringstream charFile;
			Mat charImage, featureMat;
			
			charFile << charPath << characters[i] << "/" << j << ".jpg";
			charImage = imread(charFile.str(), 0);
			featureMat = getFeatures(charImage);
			
			//字符识别
			Mat classesMat;

			classesMat.create(1, CHARNUMBER, CV_32FC1);
			annClassifier.predict(featureMat, classesMat);
			
			//字符转换
			Point maxLoc;
			
			minMaxLoc(classesMat, 0, 0, 0, &maxLoc);
			
			if(maxLoc.x == i){
				charsSample[i]++;
			}
		}
	}
	
	//显示结果
	float charsRatio[CHARNUMBER];
	
	for(int i = 0; i < CHARNUMBER; i++){
		charsRatio[i] = ( (float)charsSample[i] / (charNumber[i] - (int)(charNumber[i] * TRAINSAMPLE)) ) * 100;
		cout << characters[i] << " recongnition ratio is " << charsRatio[i] << "%\n";
	}
	
	return 0;
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