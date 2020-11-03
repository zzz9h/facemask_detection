#include <opencv2/opencv.hpp>
#include<vector>
#include<stdlib.h>
#include<string>
#include<iostream>
#include<fstream>
#include<string>
using namespace std;
using namespace cv;
using namespace cv::dnn;
bool faceDetected(Mat inputImg, Mat& outputFace, Rect &facebox, Net face_detector)
{
	
	Mat frame = inputImg;
	Mat inputBlob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false);

	face_detector.setInput(inputBlob);
	Mat prob = face_detector.forward();

	Mat detection(prob.size[2], prob.size[3], CV_32F, prob.ptr<float>());
	float confidence_thresh = 0.5;
	for (int row = 0; row < detection.rows; row++)
	{
		float confidence = detection.at<float>(row, 2);
		if (confidence > confidence_thresh)
		{
			int classID = detection.at<float>(row, 1);
			int notKnown = detection.at<float>(row, 0);
			int top_left_x = detection.at<float>(row, 3) * frame.cols;
			int top_left_y = detection.at<float>(row, 4) * frame.rows;
			int button_right_x = detection.at<float>(row, 5) * frame.cols;
			int button_right_y = detection.at<float>(row, 6) * frame.rows;
			int width = button_right_x - top_left_x;
			int height = button_right_y - top_left_y;
			Rect box(top_left_x, top_left_y, width, height);
			cout << classID << "," << notKnown << "," << confidence << endl;
			if (box.x < 0 || box.y < 0)
			{
				box.x = 0;
				box.y = 0;
			}
			else if (box.br().x > frame.cols || box.br().y > frame.rows)
			{
				box.width = frame.cols - box.x;
				box.height = frame.rows - box.y;
			}
			else if (box.x + box.width > frame.cols)
			{
				box.width = frame.cols - box.x - 1;
			}
			else if (box.y + box.height > frame.rows)
			{
				box.height = frame.rows - box.y - 1;
			}
			else if (0 < box.width && 0 < box.height)
			{
				outputFace = frame(box).clone();
				facebox = box;
			}
		}
	}
	if (outputFace.empty())
	{
		return false;
	}
	else
	{
		return true;
	}
}
bool face_mask_detectd(Mat faceImg, Ptr<ml::SVM> model)
{
	resize(faceImg, faceImg, Size(64, 128));
	Mat face_gray;
	cvtColor(faceImg, face_gray, COLOR_BGR2GRAY);
	HOGDescriptor* hog = new HOGDescriptor;
	vector<float> descriptors;
	hog->compute(face_gray, descriptors);
	float detection = model->predict(descriptors);
	if (detection > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}


void FaceMaskDetect(Mat& inputImg, Ptr<ml::SVM> detecModel, Net model)
{
	Mat face;
	Rect faceBox;
	if (faceDetected(inputImg, face, faceBox, model))
	{
		if (face_mask_detectd(face, detecModel))
		{
			rectangle(inputImg, faceBox, Scalar(0, 255, 0), 1, 8);
			string output = "Face Mask";
			putText(inputImg, output, Point(faceBox.br().x / 2, faceBox.br().y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 1, 8);
		}
		else
		{
			rectangle(inputImg, faceBox, Scalar(0, 0, 255), 1, 8);
			string output = "Not Face Mask";
			putText(inputImg, output, Point(faceBox.x, faceBox.y), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1, 8);
		}
	}
}
int main()
{
    	/************************************实时检测********************************************/
	string model_path = "opencv_face_detector_uint8.pb";
	string config_path = "opencv_face_detector.pbtxt";
	cout<<"1_called"<<endl;
	Net face_detector = readNetFromTensorflow(model_path, config_path);
	 face_detector.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	 face_detector.setPreferableTarget(DNN_TARGET_CPU);
	 cout<<"2_called"<<endl;
	 auto detecModel = ml::SVM::load("face_mask_detection.xml");
	 VideoCapture capture;
	 capture.open(0);
	 if (!capture.isOpened())
	 {
	 	cout << "can't open camera" << endl;
	 	exit(-1);
	 }
	 Mat frame;
	 cout<<"3_called"<<endl;
	 while (capture.read(frame))
	 {
	 	FaceMaskDetect(frame, detecModel, face_detector);
	 	 namedWindow("test_image", WINDOW_FREERATIO);
	 	imshow("test_image", frame);

	 	char ch = waitKey(1);
	 	if (ch == 27)
	 	{
	 		break;
	 	}
	 }
	waitKey(0);
	return 0;
}