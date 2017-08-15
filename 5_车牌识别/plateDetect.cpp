/********************************************************************
*����:plateDetect.cpp
*����:D
*ʱ��:2016.01.22
*����:���Ƽ��
*********************************************************************/

/********************************************************************
*ͷ�ļ�
*********************************************************************/
#include "plateDetect.h"

/********************************************************************
*����:plateDetect
*����:
*	sourceImage   ԴͼƬ
*����:
*	plates   ����
*����:���Ƽ��
*********************************************************************/
vector<PLATE> plateDetect(Mat sourceImage){
	//�Ҷȱ任
	Mat grayImage;
	
	cvtColor(sourceImage, grayImage, CV_BGR2GRAY);
	
	//��ֵ�˲�
	blur(grayImage, grayImage, Size(5, 5));
	
	//��ֱ���
	Sobel(grayImage, grayImage, CV_8U, 1, 0, 3);
	
	//��ֵ�任
	threshold(grayImage, grayImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	
	//��̬�任
	Mat element;
	
	element = getStructuringElement(MORPH_RECT, Size(PLATE_MORP_X, PLATE_MORP_Y));
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, element);
	
	//��ȡ����
	vector<vector<Point> > contours;
	
	findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//�����ü�
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
		//��ˮ���
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
			//��ԭ�㸽��ȡ�������
			seed.x = (rects[i].center.x) + (rand()%(int)length) - (length/2);
			seed.y = (rects[i].center.y) + (rand()%(int)length) - (length/2);
			
			floodFill(sourceImage, mask, seed, newVal, &comp, loDiff, upDiff, flags);
		}
		
		//��ȡ��ͨ��
		Mat_<uchar>::iterator maskItr;
		vector<Point> pointInterest;

		maskItr = mask.begin<uchar>();
		while( maskItr != mask.end<uchar>() ){
			if(*maskItr == 255){
				pointInterest.push_back(maskItr.pos());
			}
			
			maskItr++;
		}
		
		//�ü���ͨ��
		RotatedRect minRect;

		minRect = minAreaRect(Mat(pointInterest));
		if(!verifyPlate(minRect)){
			continue;
		}
		
		//���þ���
		Point2f rectCenter;
		Size2f rectSize;
		float rectAngle;
		
		rectCenter = minRect.center;
		rectSize = minRect.size;
		rectAngle = minRect.angle;
		
		if(minRect.size.width < minRect.size.height){   //������ο�С�ڸߣ���ת90��
			swap(rectSize.width, rectSize.height);
			rectAngle = rectAngle + 90;
		}
		
		//��תͼƬ
		Mat rotateMat;
		Mat rotateImage;
		
		rotateMat = getRotationMatrix2D(rectCenter, rectAngle, 1);
		warpAffine(sourceImage, rotateImage, rotateMat, sourceImage.size(), INTER_CUBIC);
		
		//�ü�ͼƬ
		getRectSubPix(rotateImage, rectSize, rectCenter, rotateImage);
		
		//������С
		Mat resizeImage;
		
		resizeImage.create(PLATEHEIGHT, PLATEWIDTH, CV_8UC3);
		resize(rotateImage, resizeImage, resizeImage.size(), 0, 0, INTER_CUBIC);
		
		//��һͼƬ
		Mat histImage;
		
		cvtColor(resizeImage, histImage, CV_BGR2GRAY);
		blur(histImage, histImage, Size(3, 3));
		equalizeHist(histImage, histImage);
		
 		//����֧������
		CvSVM svmClassifier;
	
		svmClassifier.load("svm.xml");
		
		//��ȡ��������
		Mat featureMat;
		
		featureMat = histImage.reshape(1, 1);
		featureMat.convertTo(featureMat, CV_32FC1);
		
		//���Ƽ��
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
*����:verifyPlate
*����:
*	rect   ��������
*����:
*	flag   1 ����
*	       0 ������
*����:��������Ƿ����㳵������
*********************************************************************/
bool verifyPlate(RotatedRect rect){
	//���þ��ο��
	float width, height;
	
	width = rect.size.width;
	height = rect.size.height;
	
	//��߱������
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
	
	//�����С���
	float platArea, miniArea, maxiArea;
	
	platArea = width * height;
	miniArea = (PLATEASPECT * 15) * 15;
	maxiArea = (PLATEASPECT * 125) * 125;
	
	if(platArea < miniArea || maxiArea < platArea){
		return false;
	}
	
	return true;
}