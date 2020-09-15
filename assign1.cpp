#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "TrackingObjectSet.h"
#include "Tracker.h"
#include <vector>

using namespace cv;
using namespace std;
int MAX_KERNEL_LENGTH1 = 15;
Mat frame;
Mat dst;
Mat img;
int thresh = 100;
int max_thresh = 255;
int Upper = 0;
int Lower = 0;
RNG rng(12345);
Mat morpho_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat morpho_kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
Tracker tracker = Tracker();
void thresh_callback(int, void*);

int main(int argc, char** argv)
{
	//Mat frame;
	Mat background;
	Mat object;
	Mat src, src_gray;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	VideoCapture cap("C:\\Users\\Boonyanut\\Pictures\\video.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	cap.read(frame);
	cap.read(img);
	Mat acc = Mat::zeros(frame.size(), CV_32FC3);
	namedWindow("frame", 1);

	for (;;)
	{
		// count person
		cap >> img;
		cap >> frame;
		if (frame.empty()) {
			//vid.release();
			cout << "Finished!" << endl;
			break;
		}

		Mat gray;
		cap >> frame; // get a new frame from camera
		imshow("frame", frame);
		// Get 50% of the new frame and add it to 50% of the accumulator
		//equalizeHist(frame, dst);
		threshold(frame, frame, 5, 255, THRESH_BINARY);
		imshow("eee", frame);
		//imshow("Threshold1", frame);
		GaussianBlur(frame, frame, Size(21, 21), 0, 0);

		//imshow("blur", frame);
		accumulateWeighted(frame, acc, 0.5);
		// Scale it to 8-bit unsigned
		convertScaleAbs(acc, background);
		subtract(frame, background, frame);
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		//absdiff(frame, background, frame);
		threshold(frame, frame, 10, 255, THRESH_BINARY);
		imshow("Threshold2", frame);
		morphologyEx(frame, frame, MORPH_OPEN, morpho_kernel); //erosion
		dilate(frame, frame, morpho_kernel, Point(3, 3), 2);

		///thresh_callback(0, 0);
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		//imshow("Threshold", grad);

		//เรียกฟังก์ชัน _thresh_callback
		thresh_callback(0, 0);
		if (waitKey(25) >= 0) break;

	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Rect> blobs;
	/// Detect edges using Threshold
	threshold(frame, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours 
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	//vector<Rect> blobs;
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		if (boundRect[i].width > 50 && boundRect[i].height > 80)
			blobs.push_back(boundRect[i]);
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}
	/// Draw polygonal contour + bonding rects + circles
	//Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing = img;
	line(drawing, Point(0, drawing.rows / 3), Point(frame.cols, frame.rows / 3), Scalar(0, 0, 0), 2);

	for (int i = 0; i < blobs.size(); i++)
	{
		Scalar color = tracker.Track(blobs[i], drawing.rows);
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, blobs[i].tl(), blobs[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
		int begin = blobs[i].y;
		int end = begin + blobs[i].height;
		if ((begin < (drawing.rows / 3)) && (end > (drawing.rows / 3))) {
			int up_or_down = tracker.rectCounter(blobs[i]);
			if (up_or_down == 1) {
				Upper++;
			}
			else if (up_or_down == -1) {
				Lower++;
			}
		}

	}
	cout << tracker.objsSet->objs.size() << endl;
	putText(drawing, "Total up: ", Point(20, 425), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
	putText(drawing, to_string(Upper), Point(150, 425), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
	putText(drawing, "Total down: ", Point(20, 455), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
	putText(drawing, to_string(Lower), Point(180, 455), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);

	/// Show in a window
	//namedWindow("Contours", 1);
	imshow("Contours", img);
}