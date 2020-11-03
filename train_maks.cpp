#include<opencv.hpp>

#include<vector>

#include<stdlib.h>

#include<string>

#include<iostream>

#include<fstream>

#include"face.h"

#include"face_mask.h"

using namespace std;

using namespace cv;

using namespace cv::dnn;

void trainSVM(vector<Mat> positive_images, vector<Mat> negative_images, string path, Net model)
{
	//分别获取正负样本中每张图像的HOG特征描述子，并进行标注
	HOGDescriptor* hog_train = new HOGDescriptor;
	vector<vector<float>> train_descriptors;
	int positive_num = positive_images.size();
	int negative_num = negative_images.size();
	vector<int> labels;
	for (int i = 0; i < positive_num; i++)
	{
		Mat positive_face;
		Rect positive_faceBox;
		if (faceDetected(positive_images[i], positive_face, positive_faceBox, model))
		{
			resize(positive_face, positive_face, Size(64, 128));
			Mat gray;
			cvtColor(positive_face, gray, COLOR_BGR2GRAY);			//计算HOG描述子时需要使用灰度图像
			vector<float> descriptor;
			hog_train->compute(gray, descriptor);
			train_descriptors.push_back(descriptor);
			labels.push_back(1);
		}
	}
	for (int j = 0; j < negative_num; j++)
	{
		Mat negative_face;
		Rect negative_faceBox;
		if (faceDetected(negative_images[j], negative_face, negative_faceBox, model))
		{
			resize(negative_face, negative_face, Size(64, 128));
			Mat gray;
			cvtColor(negative_face, gray, COLOR_BGR2GRAY);
			vector<float> descriptor;
			hog_train->compute(gray, descriptor);
			train_descriptors.push_back(descriptor);
			labels.push_back(-1);
		}
	}
	//将训练数据vector转换为Mat对象，每一行为一个描述子，行数即为样本数
	int width = train_descriptors[0].size();
	int height = train_descriptors.size();
	Mat train_data = Mat::zeros(Size(width, height), CV_32F);
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			train_data.at<float>(r, c) = train_descriptors[r][c];
		}
	}
	auto train_svm = ml::SVM::create();
	train_svm->trainAuto(train_data, ml::ROW_SAMPLE, labels);
	train_svm->save(path);
	hog_train->~HOGDescriptor();
	train_svm->clear();
}

int main()
{
		 string model_path = "D:\\opencv_c++\\opencv_tutorial\\data\\models\\face_detector\\opencv_face_detector_uint8.pb";

	 string config_path = "D:\\opencv_c++\\opencv_tutorial\\data\\models\\face_detector\\opencv_face_detector.pbtxt";

	 Net face_detector = readNetFromTensorflow(model_path, config_path);

	 face_detector.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);

	 face_detector.setPreferableTarget(DNN_TARGET_CPU);



	 string positive_path = "D:\\opencv_c++\\opencv_tutorial\\data\\Face_Mask_Detection\\positive\\";

	 string negative_path = "D:\\opencv_c++\\opencv_tutorial\\data\\Face_Mask_Detection\\negative\\";



	 vector<string> positive_images_str, negative_images_str;

	 glob(positive_path, positive_images_str);

	 glob(negative_path, negative_images_str);



	 vector<Mat>positive_images, negative_images;

	 for (int i = 0; i < positive_images_str.size(); i++)

	 {

	 	Mat positive_image = imread(positive_images_str[i]);

	 	 resize(positive_image, positive_image, Size(64, 128));

	 	positive_images.push_back(positive_image);

	 }

	 for (int j = 0; j < negative_images_str.size(); j++)

	 {

	 	Mat negative_image = imread(negative_images_str[j]);

	 	 resize(negative_image, negative_image, Size(64, 128));

	 	negative_images.push_back(negative_image);

	 }

	 string savePath = "face_mask_detection.xml";

	 trainSVM(positive_images, negative_images, savePath, face_detector);

}
