#ifndef ____recognition__
#define ____recognition__

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

string recognize(cv::Mat licencePlate);


#endif /* defined(____recognition__) */
