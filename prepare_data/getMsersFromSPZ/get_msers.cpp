#include "Features_char.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <stack>
#include <dirent.h>
#include <stdio.h>

using namespace std;
using namespace cv;

#define FEATURECOUNT 9

int c = 0;

/* 
 * Convert number to string
 */

template <typename T>
string NumberToString ( T Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

/*
 * Draw only msers
 * Create new black&white image with MSERs
 */
Mat drawMser(vector<Point> mser, Mat image)
{
    // bounding box
    std::vector<cv::Point> contour_poly;
    cv::approxPolyDP(cv::Mat(mser), contour_poly, 1, true);
    CvRect boundRect = cv::boundingRect(cv::Mat(contour_poly));
    
    cv::Mat mserImg = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
    
    // draw MSER
    for (int j = 0; j < (int)mser.size(); j++)
    {
        cv::Point point = mser[j];
        mserImg.at<char>(point.y,point.x) = 255;
    }
    
    mserImg = cv::Mat(mserImg,boundRect);
    
    // New image with MSER - with 1px bounding
    cv::Mat newImg = cv::Mat::zeros(mserImg.rows + 2,mserImg.cols + 2,CV_8UC1);
    for (int i = 0; i < mserImg.rows; i++) {
        for (int j = 0; j < mserImg.cols; j++) {
            newImg.at<char>(i+1,j+1) = mserImg.at<char>(i,j);
        }
    }
    return newImg;
}

/**  @function Erosion  */
void erosion( cv::Mat src, cv::Mat dst, int type, float size )
{
    int erosion_type = type;
    float erosion_size = size;
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    /// Apply the erosion operation
    erode( src, dst, element );
}

/** @function Dilation */
void dilation( cv::Mat src, cv::Mat dst, int type, float size )
{
    int dilation_type = type;
    float dilation_size = size;
    
    Mat element = getStructuringElement( dilation_type,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    dilate( src, dst, element );
}

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();
    
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    
    int label_count = 2; // starts at 2 because 0,1 are used already
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 0) {
                continue;
            }
            
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            
            std::vector <cv::Point2i> blob;
            
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    
                    blob.push_back(cv::Point2i(j,i));
                }
            }
            
            blobs.push_back(blob);
            
            label_count++;
        }
    }
}




int main( int argc, char* argv[])
{
	string inputImageName;
    // Msers
    cv::vector<cv::vector<cv::Point> > msers;
    // Letters
    cv::vector<cv::vector<cv::Point> > letters;
    // SPZ
    cv::vector<cv::vector<cv::Point> > spz[7];

	// zpracovani parametru prikazove radky
	for( int i = 1; i < argc; i++){
		if( string(argv[i]) == "-d" && i + 1 < argc){
			inputImageName = argv[++i];
		} else if( string(argv[i]) == "-h"){
			cout << "Use: " << argv[0] << "  -d inputFolder "<< endl;
			return 0;
		} else {
			cerr << "Use: " << argv[0] << "  -d inputFolder "<< endl;
            return -1;
		}
	}

	// kontrola zadani parametru
	if( inputImageName.empty()) {
		cerr << "Use: " << argv[0] << "  -d inputFolder "<< endl;
		return -1;
	}
    
    // load SVM model
    CvSVM SVM;
    SVM.load("model_letters.xml");
    
    cv::vector<string> images;
    DIR *dir;
    class dirent *ent;
    
    dir = opendir(inputImageName.c_str());
    while ((ent = readdir(dir)) != NULL) {
        string file_name = ent->d_name;
        
        if (file_name[0] == '.')
        continue;
        
        string file_path = inputImageName + "/" + file_name;
        images.push_back(file_path);
    }
    closedir(dir);

    for (int i = 0; i < images.size(); i++) {
        // load image
        Mat inputImage;
        inputImage = imread(images[i], CV_LOAD_IMAGE_COLOR);

        // check image
        if( inputImage.data == NULL) {
            cerr <<  "Error: Failed to read image file \"" << images[i] << "\"." << endl ;
            return -1;
        }

        // copy original image
        cv::Mat MSERImage;
        inputImage.copyTo(MSERImage);
        
        // need one channel img
        if (MSERImage.channels() != 1)
            cv::cvtColor(MSERImage, MSERImage, CV_RGB2GRAY);
        
        threshold( MSERImage, MSERImage, 0, 255,CV_THRESH_OTSU);
        
        FindBlobs(MSERImage, msers);
        
//        cv::Mat output = cv::Mat::zeros(MSERImage.size(), CV_8UC3);
//        // Randomy color the blobs
//        for(size_t i=0; i < msers.size(); i++) {
//            unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
//            unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
//            unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
//            
//            for(size_t j=0; j < msers[i].size(); j++) {
//                int x = msers[i][j].x;
//                int y = msers[i][j].y;
//                
//                output.at<cv::Vec3b>(y,x)[0] = b;
//                output.at<cv::Vec3b>(y,x)[1] = g;
//                output.at<cv::Vec3b>(y,x)[2] = r;
//            }
//        }
//        
//        imwrite("blobs.png",output);
        
        //cv::MSER mser = cv::MSER(5,5,100,0.5,0.001,200,0.01,0,1);
        //cv::MSER mser = cv::MSER();
        //mser(MSERImage,msers);
        
        for (int i = 0; i < msers.size(); i++) {
            // get image only with mser
            cv::Mat mser = drawMser(msers[i],MSERImage);
            string name = "./";
            
            // extract features from MSER
            float sampleData[1][FEATURECOUNT];
            std::vector<float> features;
            features = extractFeaturesFromMSER(mser);
            for (int i = 0; i < FEATURECOUNT; i++) {
                sampleData[0][i] = features[i];
            }
            
            // create sample Mat for SVM
            cv::Mat sampleMat = cv::Mat(1,FEATURECOUNT,CV_32FC1,sampleData);
            
            // SVM decide if MSER is letter or not
            float response = SVM.predict(sampleMat);
            
            // MSER is letter
            if (response == 1)
            {
                name.append("out/letters/");
                name.append(NumberToString(c));
                c++;
                name.append(".png");
                imwrite(name,mser);
                
                
                //cout << "pismeno: " << name << endl;
            } else {
                name.append("out/notletters/");
                name.append(NumberToString(c));
                c++;
                name.append(".png");
                imwrite(name,mser);
                //cout << "nepismeno: " << name << endl;
            }
        }
    }
}

