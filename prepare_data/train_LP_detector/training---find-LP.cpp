#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>    
#include <sys/stat.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

/*********************** DEFINICE STRUKTUR *************************/

typedef struct{     
	string inputList;
	string lpList;
    string lpDataset;
    float maxSize;
    int percent;
} param;

/**********************************************************************/

template <typename T>
string NumberToString ( T Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

/********************* ZPRACOVÁNÍ PARAMETRŮ ***************************/

param zpracujParametry (int argc, char *argv[])
{	
    if (argc < 3){
        cout << "./main -d <image-list> -w <max-size-LP> -p <range-around-LP>" << endl;
        exit(1);
   }

    // prepare struct
	param vstup;
	vstup.inputList = "";
    vstup.lpList = "LP-list.dat";
    vstup.lpDataset = "LP-dataset";
    vstup.maxSize = 1.0f;
    vstup.percent = 20;

    // get values from input
    int c;
    extern char * optarg;
    while ((c = getopt(argc, argv, "d:w:p:h")) != -1){
        switch(c) {
            case 'd':
                vstup.inputList = optarg;
                break;
            case 'w':
                vstup.maxSize = atof(optarg);
                break;
            case 'p':
                vstup.percent = atoi(optarg);
                break;
            case 'h':
                cout << "./main -d <image-list> -w <max-size-LP> -p <range-around-LP>" << endl;
                exit(0);
            default:
                cout << "./main -d <image-list> -w <max-size-LP> -p <range-around-LP>" << endl;
                exit(1);
        }       
    }

    // control of input parameters
    if (vstup.inputList.empty()) {
        cout << "./main -d <image-list> -w <max-size-LP> -p <range-around-LP>" << endl;
        exit(1);
    }

    if ((vstup.maxSize < 0.0f) || (vstup.maxSize > 1.0f)) {
        cout << "./main -d <image-list> -w <max-size-LP> -p <range-around-LP>" << endl;
        exit(1);
    }

    int pos = vstup.inputList.rfind("/");
    string path = vstup.inputList.substr(0, pos+1);
    vstup.lpDataset = path + "LP-dataset";
    vstup.lpList = path + vstup.lpList;

    // create new directory - positive dataset
    if ( access(vstup.lpDataset.c_str(), 0 ) != 0 )
        mkdir(vstup.lpDataset.c_str(), S_IRWXU|S_IRGRP|S_IXGRP);   

    return vstup;
}

/******************* Bounding box around MSER area ********************/

CvRect getPositionOfMSER(vector<Point> mser, Mat image)
{
    // create bounding box around MSER area
    vector<Point> contour;
    approxPolyDP(Mat(mser), contour, 3, true);
    CvRect boundRect = boundingRect(Mat(contour));
    return boundRect;
}

/*********************** Show image with MSER *************************/

void showMSERimg(Mat image, vector<Point> mser, CvRect boundRect)
{
    Mat mserImg = Mat::zeros(image.rows,image.cols,CV_8UC1);

    for (int j = 0; j < (int)mser.size(); j++)
    {
        cv::Point point = mser[j];
        mserImg.at<char>(point.y,point.x) = 255;
    }
    
 //   mserImg = Mat(mserImg,boundRect);
    imshow("a",mserImg);
    waitKey(0);
}

/************************ Check proportion ****************************/

bool checkProportionOfPlate (CvRect rect)
{
    double pomer = (double)rect.width / (double)rect.height; 
    if ((pomer >  3.0) && (pomer < 5.5)) return true;
    return false;
}

bool checkMinimalSizeOfPlate (CvRect rect)
{
    if (rect.height < 8) return false;  
    if (rect.width < 38) return false;
    return true;
}

bool checkMaximalWidth (CvRect rect, int imgWidth, float maxWidth)
{
    float pomer = (float)rect.width / (float)imgWidth;

    if ((pomer > maxWidth) || (pomer < 0.05))
        return false;    

    return true;
}

/*************** draw black rectangle across the LP ********************/

Mat drawBlackRect(Mat src, CvRect boundRect)
{
    Mat neg;
    src.copyTo(neg);

    Point pt [1][4];
    pt[0][0] = Point(boundRect.x, boundRect.y);
    pt[0][1] = Point(boundRect.x + boundRect.width, boundRect.y);
    pt[0][2] = Point(boundRect.x + boundRect.width, boundRect.y + boundRect.height);
    pt[0][3] = Point(boundRect.x, boundRect.y + boundRect.height);

    const Point* ppt[1] = {pt[0]};
    int npt[] = {4};

    fillPoly( neg, ppt, npt, 1, Scalar(0,0,0), 8 );

    return neg;
}

/*************** draw black rectangle across the LP ********************/

Mat drawBoundingBox(Mat src, CvRect boundRect)
{
    Mat boundImg;
    src.copyTo(boundImg);
    rectangle(boundImg, Point(boundRect.x,boundRect.y), Point(boundRect.x+boundRect.width,boundRect.y+boundRect.height), Scalar(0,255,0), 1);
    return boundImg;
}

/******** Test if there is mountains of black pixels in LP ***********/

bool pixelAnalysis(Mat spzImg)
{
    Mat gray;
    cvtColor(spzImg, gray, CV_BGR2GRAY);
    threshold(gray, gray, 100, 255, THRESH_BINARY);

    int blackPixArray [gray.cols];
    int min = 100;
    int max = 0;

    for (int i=0; i < gray.cols; i++)
    {
        blackPixArray[i] = 0;
        for (int j=0; j < gray.rows; j++){
            if ((int)gray.at<uchar>(j,i) == 0)
                blackPixArray[i]++;
        }
        if (min > blackPixArray[i]) min = blackPixArray[i];
        if (max < blackPixArray[i]) max = blackPixArray[i];
    }

    int middle = max / 2;   
    int pom = 0;

    for (int i=2; i < gray.cols-2; i++)
    {
        int prev = blackPixArray[i-1];
        int act  = blackPixArray[i];

        if ((prev < middle) && (act >= middle))
            pom++;
    }

    if ((2 < pom) && (pom < 15))
        return true;
    else
        return false;
}

/***************** Create bigger bounding box around LP ********************/

CvRect enlargesBoundingBox (CvRect boundRect, int percent, Mat src)
{
    float border = (float)boundRect.height / 100.0 * (float)percent;
    CvRect newRect = boundRect;

    if ((boundRect.x - border) < 0){
        newRect.x = 0;
        newRect.width += boundRect.x;
    }else{
        newRect.x -= border;
        newRect.width += border;
    }
    
    if ((boundRect.y - border) < 0){
        newRect.y = 0;
        newRect.width += boundRect.y;
    }else{
        newRect.y -= border;
        newRect.height += border;
    }
    
    if ((boundRect.x + boundRect.width + border) > src.cols){
        newRect.width += (src.cols - boundRect.x - boundRect.width - 1);
    }else{
        newRect.width += border;
    }

    if ((boundRect.y + boundRect.height + border) > src.rows){
        newRect.height += (src.rows - boundRect.y - boundRect.height - 1);
    }else{
        newRect.height += border;
    }

    return newRect;
}

/******************* Save LP with specific name ***********************/

void saveImage(param par, string imgPath, Mat croppedImage, CvRect newRect)
{
    int pos = imgPath.rfind("/");
    string imgName = imgPath.substr(pos+1, imgPath.length() - pos - 1);
  
    pos = imgName.rfind(".");
    string left  = imgName.substr(0, pos);
    string right = imgName.substr(pos, imgName.length()-pos);

    string values = "-";
    values += NumberToString(newRect.x);
    values += "-";
    values += NumberToString(newRect.y);
    values += "-";
    values += NumberToString(newRect.width);
    values += "-";
    values += NumberToString(newRect.height);
  
    string finalpath = par.lpDataset+"/"+left+values+right;
    imwrite(finalpath, croppedImage);
}

/****************************** MAIN **********************************/

int main (int argc, char ** argv)
{
    // read params from stdin    
	param par = zpracujParametry(argc, argv);
    
    // open empty file for list of founded LP
    ofstream lpList;
    lpList.open(par.lpList.c_str());

    // open list with images paths 
    ifstream inList(par.inputList.c_str());
    string imgPath;
    
    // read file line by line, line = imgPath
    while(getline(inList, imgPath))
    {
        cout << imgPath << endl;
        lpList << imgPath;

        // open image 
        Mat src, gray;
        src = imread(imgPath);
        if(!src.data) {
            cout << "Could not open or find the image" << endl ;
            return 1;
        }

        // prepare MSER, switch R, G, B channels and get only one channel
        Mat MSERImage;
        src.copyTo(MSERImage);
        vector<vector<Point> > mserVec;

        if (MSERImage.channels() != 1)
            cv::cvtColor(MSERImage, MSERImage, CV_RGB2GRAY);

        // detect MSER
        MSER mser = cv::MSER();
        mser(MSERImage, mserVec);
        
        // declaration 
        Mat croppedImage, boundImg, negImg;
        src.copyTo(boundImg);
        src.copyTo(negImg);

        // investigation of MSER areas
        for (int i = 0; i < mserVec.size(); i++)
        {
            // get position of MSER
            CvRect boundRect = getPositionOfMSER(mserVec[i],MSERImage);
            //showMSERimg(MSERImage, mserVec, boundRect);
    
            // check proportion of potential LP
            if (!checkProportionOfPlate(boundRect)) continue;
            if (!checkMinimalSizeOfPlate(boundRect)) continue;
            if (!checkMaximalWidth(boundRect, src.cols, par.maxSize)) continue;

            // check mountain of black pixels in potential LP
            croppedImage = src(boundRect);

            if (!pixelAnalysis(croppedImage)) continue; 

            // create bigger bounding box around LP
            CvRect newRect = enlargesBoundingBox(boundRect, par.percent, src);
            croppedImage = src(newRect);
     
            lpList << " " << newRect.x << " " << newRect.y << " " << newRect.width << " " << newRect.height;
            saveImage(par, imgPath, croppedImage, newRect);
     
            // Draw bounding box around possible LP
            //boundImg = drawBoundingBox(boundImg, newRect);

            // Draw black rectangle across LP
            //negImg = drawBlackRect(negImg, newRect);
        }

        lpList << endl;

        //imshow("srcBoundImg",boundImg);
        //imshow("neg", negImg);
        //waitKey(0);
    }
 
    lpList.close();

    return 0;
}
