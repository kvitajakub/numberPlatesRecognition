#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <fstream>   
 
using namespace std;
using namespace cv;

/*******************************************************************/

Mat cropLP(Mat src)
{
    Mat potImg, grayImg, threshImg, outImg;
    src.copyTo(potImg);
    src.copyTo(outImg);

    cvtColor(potImg, grayImg, CV_BGR2GRAY);
    equalizeHist(grayImg, grayImg);
    threshold(grayImg, threshImg, 80, 255, THRESH_BINARY);
 
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
 
    // Find contours
    findContours(threshImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
 
    // Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
        minRect[i] = minAreaRect( Mat(contours[i]) );

    // Cut only LP in picture
    for( int i = 0; i < contours.size(); i++ )
    { 
        if ((minRect[i].angle > 10) || (minRect[i].angle < -10)) continue;
        if ((minRect[i].size.width < 15) || (minRect[i].size.height < 5)) continue;
 
        int pomer = minRect[i].size.width / minRect[i].size.height;
        if ((pomer < 3.7) || (pomer > 5.7)) continue;
 
        CvRect rect = minRect[i].boundingRect();
        
        if (rect.x+rect.width  > potImg.cols) rect.width  = (rect.x+rect.width)  - potImg.cols;  
        if (rect.y+rect.height > potImg.rows) rect.height = (rect.y+rect.height) - potImg.rows; 

        outImg = src(rect);

        //rectangle(potImg, rect, Scalar(255,0,255));
        //imshow("bounding rectanglea", potImg);
        //imshow("asdfj", outImg);
        //waitKey(0);
    }

    return outImg;
}

/*******************************************************************/

cv::vector<cv::Mat> processImage(string inputImg, CascadeClassifier cascade)
{
    // licence plates images
    cv::vector<cv::Mat> platesImages;
    
    // read input image
    Mat src;
    src = imread(inputImg);
    if (src.data == NULL){
        cout << "Could not load image." << endl;
        exit(1);
    }
 
    // convert src image to grayscale 
    Mat grayImg;
    cvtColor(src, grayImg, CV_BGR2GRAY);

    // vector for founded objects
    std::vector<Rect> platesRects;

    // detection licence plates
    cascade.detectMultiScale(grayImg, platesRects, 1.1, 3, 0, Size(50, 12));
 
    // draw cropped image
    Mat croppedImg;

    for (int i = 0; i < platesRects.size(); i++)
    {
        Mat potImg; 
        potImg = src(platesRects[i]);
        croppedImg = cropLP(potImg);

        //imshow("cropped", croppedImg);
        //waitKey(0);
        
        platesImages.push_back(croppedImg);
    }    
    
    return platesImages;
}


/***************************** DETECTION ********************************/

cv::vector<cv::Mat> detect(string inputImg) {
    // load LP cascade detector (.xml file)
    CascadeClassifier cascade;
    cascade.load("cascade-final.xml");
    if (cascade.empty()){
        cout << "Could not load cascade classifier" << endl;
        exit(1);
    }
    
    // process image or dataset of images
    return processImage(inputImg,cascade);
}
