/********************************************************************
*����:charsDetect.cpp
*����:D
*ʱ��:2016.01.25
*����:�ַ�ʶ��
*********************************************************************/

/********************************************************************
*ͷ�ļ�
*********************************************************************/
#include "charsDetect.h"

/********************************************************************
*����:charsDetect
*����:
*	sourceImage   ԴͼƬ
*����:
*	plateNumber   ���ƺ�
*����:���Ƽ��
*********************************************************************/
string charsDetect(Mat sourceImage){
	//��ֵ�仯
	Mat thresholdImage;
	
	threshold(sourceImage, thresholdImage, 200, 255, CV_THRESH_BINARY);   //�ַ���ɫ��Ϊ��ɫ�������������
	
	//��ȡ����
	Mat contoursImage;
	vector<vector<Point> > contours;
	
	thresholdImage.copyTo(contoursImage);
	findContours(contoursImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	//�����ü�
	vector<vector<Point> >::iterator contoursItr;
	vector<CHARS> chars;
	
	contoursItr = contours.begin();
	while( contoursItr != contours.end() ){
		Rect minRect;

		minRect = boundingRect(Mat(*contoursItr));
		Mat rect(thresholdImage, minRect);
		if(verifyChar(rect)){
			//λ�õ���
			float width, height, length;
			Mat transformMat, warpImage;
			
			width = rect.cols;
			height = rect.rows;
			length = (height > width) ? height : width;
			
			transformMat = Mat::eye(2, 3, CV_32FC1);
			transformMat.at<float>(0, 2) = (length - width)/2;   //������ͼ������
			transformMat.at<float>(1, 2) = (length - height)/2;
			
			warpImage.create(length, length, rect.type());
			warpAffine(rect, warpImage, transformMat, warpImage.size());
			
			//������С
			Mat resizeImage;
			
			resizeImage.create(CHARLENGTH, CHARLENGTH, warpImage.type());
			resize(warpImage, resizeImage, resizeImage.size());
			
 			//����������
			CvANN_MLP annClassifier;
		
			annClassifier.load("ann.xml");
			
			//��ȡ��������
			Mat featureMat;
			
			featureMat = getFeatures(resizeImage);
			
			//�ַ�ʶ��
			Mat classesMat;

			classesMat.create(1, CHARNUMBER, CV_32FC1);
			annClassifier.predict(featureMat, classesMat);
			
			//�ַ�ת��
			Point maxLoc;
			
			minMaxLoc(classesMat, 0, 0, 0, &maxLoc);
			
			//�ַ�����
			CHARS temp;
			
			temp.cha = characters[maxLoc.x];
			temp.pos = minRect.x;
			
			chars.push_back(temp);
		}

		contoursItr++;
	}
	
	//��ֵ����
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
	
	//�ַ�����
	string plateNumber;
	
 	for(int i = 1; i < chars.size(); i++){   //��Ϊ�޷���ʾ���֣�����ȥ����һ���ַ�
		plateNumber += chars[i].cha;
	}
	
	return plateNumber;
}

/********************************************************************
*����:verifyChar
*����:
*	rect   ��������
*����:
*	flag   1 ����
*	       0 ������
*����:��������Ƿ������ַ�����
*********************************************************************/
bool verifyChar(Mat rect){
	//���þ��ο��
	float width, height;

	width = rect.cols;
	height = rect.rows;
	
	//��߱������
	float charAspect, miniAspect, maxiAspect;
	
	charAspect = width / height;
	miniAspect = MINIASPECT;
	maxiAspect = CHARASPECT + (CHARASPECT * CHARERROR);
	
	if(charAspect < miniAspect || maxiAspect < charAspect){
		return false;
	}
	
	//�߶ȴ�С���
	if(height < MINCHARHEIGHT || MXNCHARHEIGHT < height){
		return false;
	}
	
	//����������
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
*����:getFeatures
*����:
*	charImage   �ַ�ͼƬ
*����:
*	featureMat   ��������
*����:��ȡ�ַ�����
*********************************************************************/
Mat getFeatures(Mat charImage){
	//ͶӰֱ��ͼ
	Mat vhist, hhist;
	
	vhist = Mat::zeros(1, charImage.cols, CV_32FC1);
	for(int i = 0; i < charImage.cols; i++){
		vhist.at<float>(i) = countNonZero(charImage.col(i));
	}
	
	hhist = Mat::zeros(1, charImage.rows, CV_32FC1);
	for(int i = 0; i < charImage.rows; i++){
		hhist.at<float>(i) = countNonZero(charImage.row(i));
	}
	
	//��һֱ��ͼ
	double maxVal;
	
	minMaxLoc(vhist, 0, &maxVal, 0, 0);
	if(maxVal > 0){
		vhist.convertTo(vhist, -1, 1.0/maxVal, 0);
	}
	
	minMaxLoc(hhist, 0, &maxVal, 0, 0);
	if(maxVal > 0){
		hhist.convertTo(hhist, -1, 1.0/maxVal, 0);
	}
	
	//�ͷֱ�������
	Mat resizeImage;
	
	resizeImage.create(LOWSLENGTH, LOWSLENGTH, charImage.type());
	resize(charImage, resizeImage, resizeImage.size());
	
	//������������
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