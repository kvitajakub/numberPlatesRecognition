#ifndef _detection_h
#define _detection_h

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdio.h>

using namespace std;
using namespace cv;

cv::vector<cv::Mat> detect(string inputImg);

#endif
