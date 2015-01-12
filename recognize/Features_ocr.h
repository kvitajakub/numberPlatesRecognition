#ifndef ____Features_ocr__
#define ____Features_ocr__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdio.h>

//#define FEATURESOCRCOUNT 224
//#define FEATURESOCRCOUNT 224
//#define FEATURESOCRCOUNT 266
//#define FEATURESOCRCOUNT 896

using namespace std;
using namespace cv;

std::vector<float> extractOCRFeatures(cv::Mat letter);

#endif /* defined(____Features_ocr__) */
