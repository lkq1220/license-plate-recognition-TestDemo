/********************************************************************
*����:main.cpp
*����:D
*ʱ��:2016.01.22
*����:����ʶ��
*********************************************************************/

/********************************************************************
*ͷ�ļ�
*********************************************************************/
#include "plateDetect.h"
#include "charsDetect.h"

/********************************************************************
*����:main
*����:
*	argc   ����
*	argv   ����
*����:
*	state    0 �ɹ�
*	        -1 ʧ��
*����:������
*********************************************************************/
int main(int argc, char **argv){
	//�������
	if(argc != 2){
		cout << "Usage:\n./PlateRecognition <image>\n";
		return -1;
	}
	
	//��ȡͼƬ
	Mat sourceImage;
	
	sourceImage = imread(argv[1]);
	
	//���Ƽ��
	vector<PLATE> plates;
	
	plates = plateDetect(sourceImage);
	
	//�ַ�ʶ��
	for(int i = 0; i < plates.size(); i++){
		plates[i].num = charsDetect(plates[i].img);
	}
	
	//��ʾ���
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