#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>    
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <string>
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
	string path;
	string inputList;
	string lpList;
	string posList;
    string negList;
    string lpDataset;
    string posDataset;
    string negDataset;
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

param zpracujParametry(int argc, char *argv[])
{	
    if (argc < 3){
        cout << "./main -d <LP-list>" << endl;
        exit(1);
   }

    // prepare struct
	param vstup;
	vstup.path = "";
	vstup.inputList = "";
    vstup.posList = "pos-list.dat";
    vstup.negList = "neg-list.dat";
    vstup.lpDataset  = "LP-dataset";
    vstup.posDataset = "positive-dataset";
    vstup.negDataset = "negative-dataset";
//    vstup.maxSize = 1.0f;
//    vstup.percent = 20;

    // get values from input
    int c;
    extern char * optarg;
    while ((c = getopt(argc, argv, "d:h")) != -1){
        switch(c) {
            case 'd':
                vstup.inputList = optarg;
                break;
            case 'h':
                cout << "./main -d <LP-list>" << endl;
                exit(0);
            default:
                cout << "./main -d <LP-list>" << endl;
                exit(1);
        }       
    }

    int pos = vstup.inputList.rfind("/");
    string path = vstup.inputList.substr(0, pos+1);

    // control of input parameters
    if (vstup.inputList.empty() || path.empty()) {
        cout << "./main -d <LP-list>" << endl;
        exit(1);
    }

    vstup.path = path;
    vstup.lpDataset = path+vstup.lpDataset;
    vstup.posDataset = path+vstup.posDataset;
    vstup.negDataset = path+vstup.negDataset;
    vstup.posList = path+vstup.posList; 
    vstup.negList = path+vstup.negList;

    // create new directory - positive dataset
    if ( access(vstup.posDataset.c_str(), 0 ) != 0 )
        mkdir(vstup.posDataset.c_str(), S_IRWXU|S_IRGRP|S_IXGRP);   

    // create new directory - negative dataset
    if ( access(vstup.negDataset.c_str(), 0 ) != 0 )
        mkdir(vstup.negDataset.c_str(), S_IRWXU|S_IRGRP|S_IXGRP);   

    return vstup;
}

/**********************************************************************/

string getImgName(string line)
{
    istringstream iss(line); 
	string imgName; 
	iss >> imgName;
    return imgName;
}

string getPathToDirectory(string imgName)
{
	string path;
	int pos = imgName.rfind("/");
	path = imgName.substr(0, pos+1);
	return path;
}

/**********************************************************************/

CvRect getPositionOfLP(string &line)
{
    CvRect rect;
    istringstream iss(line);
    iss >> rect.x >> rect.y >> rect.width >> rect.height;

    string position = NumberToString(rect.x) + " ";
    position = position + NumberToString(rect.y) + " ";
    position = position + NumberToString(rect.width) + " ";
    position = position + NumberToString(rect.height) + " ";

    int len = position.length();

    if (((int)line.length()-len) < 0)
        line = ""; 
    else
        line = line.substr(len, line.length()-len); 

    return rect;
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

/**********************************************************************/

bool checkIfLPExist(string path, string imgName, CvRect lpRect)
{
    int pos = imgName.rfind("/");
    imgName = imgName.substr(pos+1, imgName.length()-pos);

    pos = imgName.rfind(".");
    string left  = imgName.substr(0, pos); 
    string right = imgName.substr(pos, imgName.length()-pos);

    string finalPath = path + "/";
    finalPath = finalPath + left + "-" + NumberToString(lpRect.x) + "-";
    finalPath = finalPath + NumberToString(lpRect.y) + "-";
    finalPath = finalPath + NumberToString(lpRect.width) + "-";
    finalPath = finalPath + NumberToString(lpRect.height) + right;

    Mat tryImg = imread(finalPath);
    if( tryImg.data == NULL) 
        return false;
    else    
        return true;
}
/**********************************************************************/

void saveImage(string pathToDataset, string imgName, Mat img)
{
    int pos = imgName.rfind("/");
    imgName = imgName.substr(pos, imgName.length() - pos);

    string finalPath = pathToDataset + imgName;

    imwrite(finalPath, img);
}

/**********************************************************************/

string getNameOfImage (string path)
{
    string name;
    int pos = path.rfind("/");
    name = path.substr(pos+1, path.length() - pos);
    return name;
}

/****************************** MAIN **********************************/

int main (int argc, char ** argv)
{
    // read params from stdin    
	param par = zpracujParametry(argc, argv);

    // open list with images paths 
	ifstream inputList(par.inputList.c_str());   
    string line;

    // open empty file for list of positive and negative images
    ofstream posList;
    posList.open(par.posList.c_str());
    ofstream negList;
    negList.open(par.negList.c_str());

    // loop across all images from input file
    while(getline(inputList, line))
    {
        // get path to origin image and path to directory
        string imgName = getImgName(line);
        string path = getPathToDirectory(imgName);
        cout << imgName << endl;

		// is a spz in image
		int pos = line.find(" ");
		if (pos == -1) continue;

        // get only positions of LP in this line
		line = line.substr(pos+1, line.length() - pos);
        
        // declaration of necessary matrixes
        Mat src, boundImg, negImg;
        src = imread(imgName);
        src.copyTo(boundImg);
        src.copyTo(negImg);

        // create vector of rectangles
        vector<CvRect> rectVect;
        vector<CvRect>::iterator it;

        while(!line.empty()){
            // get position of licence plate in image
            CvRect lpRect = getPositionOfLP(line);
        
            if (!checkIfLPExist(par.lpDataset, imgName, lpRect)) continue;
        
            // add rectangle to vector
            it = rectVect.end();
            it = rectVect.insert(it, lpRect);

            // Draw bounding box around possible LP
            boundImg = drawBoundingBox(boundImg, lpRect);
       
            // Draw black rectangle across LP
            negImg = drawBlackRect(negImg, lpRect);
        }

        // write in negative list
        negList << "negative-dataset/" << getNameOfImage(imgName) << endl;

        // write in positive list
        if (rectVect.size() > 0){                    
            posList << "positive-dataset/" << getNameOfImage(imgName) << " ";
            posList << rectVect.size() << " ";
            for (it=rectVect.begin(); it<rectVect.end(); it++)
                posList << (*it).x << " " << (*it).y << " " << (*it).width << " " << (*it).height << " ";
            posList << endl;
        }
         
        saveImage(par.posDataset, imgName, src);
        saveImage(par.negDataset, imgName, negImg);
    }

    inputList.close();
    posList.close();
    negList.close();

    return 0;
}
