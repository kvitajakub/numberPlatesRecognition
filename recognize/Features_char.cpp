#include "Features_char.h"

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


float eulerNumber;
float perimeter;
int mserArea;
float convexHullArea;
float holesCount;
int holesArea;
int crossOneSixth;
int crossHalf;
int crossFiveSixth;

int inMSER = -1;
int notInMSER = 0;

// first feature
float widthToHeigthRatio(cv::Mat image) {
    float x = image.cols-2;
    float y = image.rows-2;
    return x/y;
}

// third feature
float compactness() {
    float x = perimeter;
    float y = mserArea;
    return x*x / y;
}

// fourth feature
float holesToAreaRatio() {
    float x = holesArea;
    float y = mserArea;
    return x / y;
}

// fifth feature
float convexHullToAreaRatio() {
    float x = convexHullArea;
    float y = mserArea;
    return x / y;
}

// sixth feature
float relativeHeight(cv::Mat image) {
    float x = mserArea;
    float y = image.rows-2;
    return x / (y*y);
}

// count euler number and perimeter
void countEulerNumberAndPerimeterAndMSERArea(cv::Mat image) {
    int ones = 0;
    int zeros = 0;
    int Q1 = 0;
    int Q2 = 0;
    int Q3 = 0;
    int QD = 0;
    
    mserArea = 0;
    
    // Count number of patterns Qn in image
    // During count mser area
    for (int j = 0; j < image.rows - 1; j++) {
        for (int i = 0; i < image.cols - 1; i++) {
            // Q params
            (image.at<char>(j,i) == notInMSER) ? zeros++ : ones++;
            (image.at<char>(j+1,i) == notInMSER) ? zeros++ : ones++;
            (image.at<char>(j,i+1) == notInMSER) ? zeros++ : ones++;
            (image.at<char>(j+1,i+1) == notInMSER) ? zeros++ : ones++;
            
            if (zeros == 3) Q1++;
            if (zeros == 1) Q3++;
            if (zeros == 2) {
                if (((image.at<char>(j,i) == notInMSER) && (image.at<char>(j+1,i+1) == notInMSER)) ||
                    ((image.at<char>(j,i) == inMSER) && (image.at<char>(j+1,i+1) == inMSER)))
                {
                    QD++;
                } else Q2++;
            }
            
            ones = 0;
            zeros = 0;
            
            //MSER area
            if (image.at<char>(j,i) == inMSER) {
                //count mserArea
                mserArea++;
            }
        }
    }
    
    
    // Euler Number
    eulerNumber = (1.0/4.0) * (Q1 - Q3 + (2*QD));
    
    // Perimeter
    perimeter = Q2 + (1.0/sqrt(2.0)) * (Q1 + Q3 + 2*QD);
    
    // Holes count
    holesCount = 1 - eulerNumber;
}

// Floof fill image - count MSER surroundings - without holes
int floodFillImage(cv::Mat img, int x, int y) {
    // Stack for flood fill
    std::stack<cv::Point> stack = std::stack<cv::Point>();
    int count = 0;
    cv::Point point = cv::Point();
    point.x = x;
    point.y = y;
    // Push first point to stack
    stack.push(point);
    
    // While stack is not empty
    while (stack.size() > 0) {
        // get first point from stack
        cv::Point p = stack.top();
        stack.pop();
        
        // If it is not inside MSER
        if (img.at<char>(p.y,p.x) == notInMSER) {
            // Count it to MSER surroundings area
            count++;
            // Switch point value
            img.at<char>(p.y,p.x) = inMSER;
            // Push surrounding points at stack
            stack.push(cv::Point(p.x+1,p.y));
            stack.push(cv::Point(p.x-1,p.y));
            stack.push(cv::Point(p.x,p.y+1));
            stack.push(cv::Point(p.x,p.y-1));
        }
    }
    
    return count;
}

// count holes in mser area
void countHolesArea(cv::Mat mser)
{
    // create frame for flood fill
    cv::Mat newImg = cv::Mat::ones(mser.rows + 2,mser.cols + 2,CV_8UC1);
    for (int i = 1; i < newImg.rows - 1; i++) {
        for (int j = 1; j < newImg.cols - 1; j++) {
            newImg.at<char>(i,j) = mser.at<char>(i-1,j-1);
        }
    }
    // count surrounding area
    int surroundingsArea = floodFillImage(newImg,1,1);
    
    // count holes area
    holesArea = mser.rows * mser.cols - surroundingsArea - mserArea;
}

// Count crosses with mser
int countCrossesAt(int row, cv::Mat image) {
    bool isInMSER = false;
    int cross = 0;
    
    for (int i = 0; i < image.cols; i++) {
        if ((image.at<char>(row,i) == inMSER) && (!isInMSER)) {
            cross++;
            isInMSER = true;
        }
        if ((image.at<char>(row,i) == notInMSER) && (isInMSER)) {
            cross++;
            isInMSER = false;
        }
    }
    return cross;
}

// Count crosses in image with mser
void countCountOfCrossesInImage(cv::Mat image) {
    int oneSixth = 1.0/6.0 * (image.rows-2);
    int half = (image.rows-2) / 2.0;
    int fiveSixth = 5.0/6.0 * (image.rows-2);
    
    crossOneSixth = countCrossesAt(oneSixth,image);
    crossHalf = countCrossesAt(half,image);
    crossFiveSixth = countCrossesAt(fiveSixth,image);
}

// count convex hull of mser area
void convexHull(cv::Mat image) {
    cv::vector<cv::Point> mser = cv::vector<cv::Point>(mserArea);
    int k = 0;
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            if (image.at<char>(j,i) == inMSER) {
                mser[k] = cv::Point(i,j);
                k++;
            }
        }
    }
    // Convex hull points
    cv::vector<cv::Point> covexHull;
    
    cv::convexHull(mser, covexHull);
    // Approximating polygonal curve to convex hull
    cv::vector<cv::Point> contour;
    approxPolyDP(cv::Mat(covexHull), contour, 0.001, true);
    
    //std::cout << std::endl << "CONVEX:" << contourArea(cv::Mat(contour)) << mser.size();
    // count convex hull Area
    convexHullArea = contourArea(cv::Mat(contour));
}

/*
 * Extract features from MSER
 */
std::vector<float> extractFeaturesFromMSER(cv::Mat mser) {
    std::vector<float> features(9);
    
    countEulerNumberAndPerimeterAndMSERArea(mser);
    countHolesArea(mser);
    countCountOfCrossesInImage(mser);
    convexHull(mser);
    
    features[0] = widthToHeigthRatio(mser);
    features[1] = holesCount;
    features[2] = compactness();
    features[3] = holesToAreaRatio();
    features[4] = convexHullToAreaRatio();
    features[5] = relativeHeight(mser);
    features[6] = crossOneSixth;
    features[7] = crossHalf;
    features[8] = crossFiveSixth;
    
    return features;
}

