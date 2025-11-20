#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"text_det.h"
#include"text_angle_cls.h"
#include"text_rec.h"

using namespace cv;
using namespace std;
using namespace Ort;


int main()
{
	TextDetector detect_model;
	// TextClassifier angle_model;
	TextRecognizer rec_model;

	// string imgpath = "/Users/zhangxin/github/PaddleOCR-v3-onnxrun-cpp-py/cpp/images/1.jpg";
	string imgpath = R"(D:\github\PaddleOCR-v3-onnxrun-cpp-py\cpp_win\images\1.jpg)";
	cout << imgpath <<endl;
	Mat srcimg = imread(imgpath);
	///cv::rotate(srcimg, srcimg, 1);
	cout << "image: " << imgpath << ", img.shape:" << srcimg.rows << "," << srcimg.cols << endl;

	vector< vector<Point2f> > results = detect_model.detect(srcimg);
	cout << "size of boxes : " << results.size() << endl;
	for (size_t i = 0; i < results.size(); i++)
	{
		cout << i << " : ";
		cout << results[i][0] << "," << results[i][1] << "; ";
		cout << results[i][2] << "," << results[i][3] << "; ";
		cout << results[i][4] << "," << results[i][5] << "; ";
		cout << results[i][6] << "," << results[i][7] << "; ";
		cout << endl;
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);
		/*imshow("textimg", textimg);
		waitKey(0);
		destroyAllWindows();*/
		/*
		if (angle_model.predict(textimg) == 1)
		{
			cv::rotate(textimg, textimg, 1);
		}
		*/
		cout << "text image: " << textimg.rows << "," << textimg.cols << endl;
		string text = rec_model.predict_text(textimg);
		cout << text << endl;
	}
	detect_model.draw_pred(srcimg, results);
	
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
	//imwrite("result.jpg", srcimg);
}