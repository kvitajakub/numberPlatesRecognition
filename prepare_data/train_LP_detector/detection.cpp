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
 
/*********************** DEFINICE STRUKTUR *************************/

typedef struct{     
    string inputImg;
    string inputList;
    string detector;
} param;

/*******************************************************************/

template <typename T>
string NumberToString ( T Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

/********************* ZPRACOVÁNÍ PARAMETRŮ ************************/

param zpracujParametry(int argc, char *argv[])
{   
    // prepare struct
    param vstup;
    vstup.inputImg = "";
    vstup.inputList = "";
    vstup.detector = "";
 
    // get values from input
    int c;
    extern char * optarg;
    while ((c = getopt(argc, argv, "i:l:d:h")) != -1){
        switch(c) {
            case 'i':
                vstup.inputImg = optarg;
                break;
            case 'd':
                vstup.detector = optarg;
                break;            
            case 'l':
                vstup.inputList = optarg;
                break;        
            case 'h':
                cout << "./main -i <src> -l <list> -d <detector>" << endl;
                exit(0);
            default:
                cout << "./main -i <src> -l <list> -d <detector>" << endl;
                exit(1);
        }       
    }

    // control of input parameters
    if (vstup.detector.empty()) {
        cout << "./main -i <src> -l <list> -d <detector>" << endl;
        exit(1);
    }

    return vstup;
}

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
        if (rect.x < 0) rect.x = 0;
        if (rect.y < 0) rect.y = 0;
        if (rect.width  > potImg.cols) rect.width  = potImg.cols;
        if (rect.height > potImg.rows) rect.height = potImg.rows;

        //rectangle(potImg, rect, Scalar(255,0,255));
        //imshow("bounding rectanglea", potImg);
        //imshow("asdfj", outImg);
        //waitKey(0);
    }

    return outImg;
}

/*******************************************************************/

void processImage(param par, CascadeClassifier cascade)
{
    // read input image
    Mat src;
    src = imread(par.inputImg);  
    if (src.data == NULL){
        cout << "Could not load image." << endl;
        exit(1);
    }
    //imshow("src", src);
 
    // convert src image to grayscale 
    Mat grayImg;
    cvtColor(src, grayImg, CV_BGR2GRAY);

    // vector for founded objects
    std::vector<Rect> plates;

    // detection licence plates
    cascade.detectMultiScale(grayImg, plates, 1.1, 3, 0, Size(50, 12));
 
    // draw rectangle on the detected plates
    Mat boundImg, croppedImg;
    src.copyTo(boundImg);

    for (int i = 0; i < plates.size(); i++)
    {
        rectangle(boundImg, Point(plates[i].x,plates[i].y), Point(plates[i].x+plates[i].width,plates[i].y+plates[i].height), Scalar(0,255,0), 2);
        
        Mat potImg; 
        potImg = src(plates[i]);
        croppedImg = cropLP(potImg);

        imshow("detected LP", boundImg );
        imshow("cropped", croppedImg);
        waitKey(0);  
    }    
    //imshow("detected LP", boundImg );
    //waitKey(0);  
}

/*******************************************************************/

string getPath (string path)
{
    string name;
    int pos = path.rfind("/");
    name = path.substr(0, pos);
    return name;
}

string getName (string path)
{
    string name;
    int pos = path.rfind("/");
    name = path.substr(pos+1, path.length() - pos);
    pos = name.find(".");
    name = name.substr(0,pos);
    return name;
}

/*******************************************************************/

void processList(param par, CascadeClassifier cascade)
{
    int posLP = 0;
    int negLP = 0;

    // open list with images paths 
    ifstream inputList(par.inputList.c_str());   
    string line;

    // loop across all images from input file
    while(getline(inputList, line))
    {
        cout << line << endl;
        string path = getPath(line);
        string name = getName(line);

        Mat src;
        src = imread(line);  
        if (src.data == NULL){
            cout << "Could not load image." << endl;
            exit(1);
        }

        // convert src image to grayscale 
        Mat grayImg;
        cvtColor(src, grayImg, CV_BGR2GRAY);

        // vector for founded objects
        std::vector<Rect> plates;

        // detection licence plates
        cascade.detectMultiScale(grayImg, plates, 1.1, 3, 0, Size(50, 12));

        // draw rectangle on the detected plates
        Mat boundImg, croppedImg;
        src.copyTo(boundImg);

        int poc = 0;
        
        for (int i = 0; i < plates.size(); i++)
        {
            //src.copyTo(boundImg);
            rectangle(boundImg, Point(plates[i].x,plates[i].y), Point(plates[i].x+plates[i].width,plates[i].y+plates[i].height), Scalar(0,255,0), 2);

            Mat potImg; 
            potImg = src(plates[i]);

            croppedImg = cropLP(potImg);
            
            //imshow("bounded", boundImg);
            //imshow("cropped", croppedImg);
            //waitKey(0);
    
            // write SPZ in directory
            //string outPath = path+"LP/";
            //outPath += name;
            //outPath += "-";
            //outPath += NumberToString(poc);
            //outPath += ".png";
            //cout << "--> " << outPath << endl; 
            //imwrite(outPath, croppedImg);
            //poc++;

            // TADY JE OBDELNIK S POZNAVACI ZNACKOU, POTAZMO SPATNE NADETEKOVANYM SUMEM 
        }    

        imshow("bounded", boundImg);
        waitKey(0);

 //       bool run = true;
 //       while(run){
 //           switch(255 & waitKey(0)){
 //               case 'j':
 //                   posLP++;
 //                   break;
 //               case 'f':
 //                   negLP++;
 //                   break;
 //               case 'm':
 //                   run = false;
 //                   break;
 //               case 'q':
 //                   cout << "positive LP: " << posLP << endl;
 //                   cout << "negative LP: " << negLP << endl;
 //                   exit(0);
 //                   break;
 //           }            
 //       }        
    }
//    cout << "positive LP: " << posLP << endl;
//    cout << "negative LP: " << negLP << endl;
}

/***************************** MAIN ********************************/

int main (int argc, char ** argv)
{
    // read params from stdin    
    param par = zpracujParametry(argc, argv);

    // load LP cascade detector (.xml file)
    CascadeClassifier cascade;
    cascade.load(par.detector);
    if (cascade.empty()){
        cout << "Could not load cascade classifier" << endl;
        exit(1);
    }

    // process image or dataset of images
    if (!par.inputImg.empty())  processImage(par, cascade);
    else if (!par.inputList.empty()) processList(par, cascade);
   
    return 0;
}