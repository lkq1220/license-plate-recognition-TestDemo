/********************************************************************
*名称:PlateTrain.c
*作者:D
*时间:2016.01.19
*功能:车牌学习
*********************************************************************/

/********************************************************************
*头文件
*********************************************************************/
#include <iostream>
#include <highgui.h>
#include <ml.h>

/********************************************************************
*宏定义
*********************************************************************/
#define TRAINSAMPLE 0.8   //训练样本比例

/********************************************************************
*命名空间
*********************************************************************/
using namespace std;
using namespace cv;

/********************************************************************
*函数原型
*********************************************************************/
int main(int argc, char **argv);

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
	if(argc != 3){
		cout << "Usage:\n./PlateTrain <Plate Number> <Other Number>\n";
		return -1;
	}
	
	//读取数据
	int plateNumber, otherNumber;
	string platePath, otherPath;

	Mat images;
	vector<float> labels;
	
	plateNumber = atoi(argv[1]);
	platePath = "img/plate/";
	for(int i = 0; i < (int)(plateNumber * TRAINSAMPLE); i++){
		stringstream plateFile;
		Mat plateImage, featureMat;
		
		plateFile << platePath << i << ".jpg";
		plateImage = imread(plateFile.str(), 0);
		featureMat = plateImage.reshape(1, 1);
		
		images.push_back(featureMat);
		labels.push_back(1);
	}
	
	otherNumber = atoi(argv[2]);
	otherPath = "img/other/";
	for(int i = 0; i < (int)(otherNumber * TRAINSAMPLE); i++){
		stringstream otherFile;
		Mat otherImage, featureMat;
		
		otherFile << otherPath << i << ".jpg";
		otherImage = imread(otherFile.str(), 0);
		featureMat = otherImage.reshape(1, 1);
		
		images.push_back(featureMat);
		labels.push_back(0);
	}
	
	//设置数据
	Mat trainingData;
	Mat trainingLabs;
	
	Mat(images).copyTo(trainingData);
	Mat(labels).copyTo(trainingLabs);
	
	trainingData.convertTo(trainingData, CV_32FC1);
	trainingLabs.convertTo(trainingLabs, CV_32FC1);
	
	//学习模型
	CvSVMParams svmParams;
	CvSVM svmClassifier;
	
	svmParams.kernel_type = CvSVM::LINEAR;
	svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);

	svmClassifier.train(trainingData, trainingLabs, Mat(), Mat(), svmParams);
	
	//保存模型
	svmClassifier.save("tmp/svm.xml");
	
	//测试模型
	int plateSample, otherSample;
	
	plateSample = 0;
	for(int i = (int)(plateNumber * TRAINSAMPLE); i < plateNumber; i++){
		//提取亮度特征
		stringstream plateFile;
		Mat plateImage, featureMat;
		
		plateFile << platePath << i << ".jpg";
		plateImage = imread(plateFile.str(), 0);
		featureMat = plateImage.reshape(1, 1);
		featureMat.convertTo(featureMat, CV_32FC1);

		//车牌检测
		float classes;
		
		classes = svmClassifier.predict(featureMat);
		if(classes == 1){
			plateSample++;
		}
	}
	
	otherSample = 0;
	for(int i = (int)(otherNumber * TRAINSAMPLE); i < otherNumber; i++){
		//提取亮度特征
		stringstream otherFile;
		Mat otherImage, featureMat;
		
		otherFile << otherPath << i << ".jpg";
		otherImage = imread(otherFile.str(), 0);
		featureMat = otherImage.reshape(1, 1);
		featureMat.convertTo(featureMat, CV_32FC1);

		//车牌检测
		float classes;
		
		classes = svmClassifier.predict(featureMat);
		if(classes == 0){
			otherSample++;
		}
	}
	
	//显示结果
	float plateRatio, otherRatio;
	
	plateRatio = ( (float)plateSample / (plateNumber - (int)(plateNumber * TRAINSAMPLE)) ) * 100;
	cout << "plate recongnition ratio is " << plateRatio << "%\n";
	
	otherRatio = ( (float)otherSample / (otherNumber - (int)(otherNumber * TRAINSAMPLE)) ) * 100;
	cout << "other recongnition ratio is " << otherRatio << "%\n";
	
	return 0;
}