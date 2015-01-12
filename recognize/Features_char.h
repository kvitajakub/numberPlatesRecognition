#ifndef _Features_char_h
#define _Features_char_h

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdio.h>

std::vector<float> extractFeaturesFromMSER(cv::Mat mser);

#endif
