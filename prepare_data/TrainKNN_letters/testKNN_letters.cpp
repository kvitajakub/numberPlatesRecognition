//
//  testSVM_mser.cpp
//

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
#include "Features_ocr.h"

using namespace std;
using namespace cv;


int main (int argc, char** argv) {
    string inputFolder;
    int trueCount = 0;
    int falseCount = 0;
    
    // zpracovani parametru prikazove radky
    for( int i = 1; i < argc; i++){
        if( string(argv[i]) == "-d" && i + 1 < argc){
            inputFolder = argv[++i];
        } else if( string(argv[i]) == "-h"){
            cout << "Use: " << argv[0] << "  -d inputFolderName "<< endl;
            return 0;
        } else {
            cerr << "Use: " << argv[0] << "  -d inputFolderName "<< endl;
            return -1;
        }
    }
    
    // kontrola zadani parametru
    if( inputFolder.empty()) {
        cerr << "Use: " << argv[0] << "  -d inputFolderName "<< endl;
        return -1;
    }
    
    cv::Mat samples;
    cv::Mat responses;
    FileStorage samplesFile("generalsamples.data",FileStorage::READ);
    FileStorage responsesFile("generalresponses.data",FileStorage::READ);
    
    samplesFile["samples"] >> samples;
    responsesFile["responses"] >> responses;
    
    /////////// LEARN KNN //////////////
    CvKNearest knn = CvKNearest();
    knn.train(samples, responses);
    
    cv::vector<string> folders;
    DIR *dir;
    class dirent *ent;
    
    dir = opendir(inputFolder.c_str());
    while ((ent = readdir(dir)) != NULL) {
        string file_name = ent->d_name;
        
        if (file_name[0] == '.')
            continue;
        
        string file_path = inputFolder + "/" + file_name;
        folders.push_back(file_path);
    }
    closedir(dir);
    
    // go throw all folders
    for (int i = 0; i < folders.size(); i++) {
        cv::vector<string> images;
        string letter = folders[i].substr(folders[i].length()-1);
        
        dir = opendir(folders[i].c_str());
        while ((ent = readdir(dir)) != NULL) {
            string file_name = ent->d_name;
            
            if (file_name[0] == '.')
                continue;
            
            string file_path = folders[i] + "/" + file_name;
            images.push_back(file_path);
        }
        closedir(dir);
        
        // go throw all images for one letter
        for (int j = 0; j < images.size(); j++) {
            // load image
            Mat inputImage;
            inputImage = imread(images[j], CV_LOAD_IMAGE_GRAYSCALE);
            
            // check image
            if( inputImage.data == NULL) {
                cerr <<  "Error: Failed to read image file \"" << images[j] << "\"." << endl ;
                return -1;
            }
            
            cv::Mat sample_mat;
            cv::vector<float> sample_vec = extractOCRFeatures(inputImage);
            
            float _l[sample_vec.size()];
            CvMat l = cvMat( 1, sample_vec.size(), CV_32FC1, _l );
            for (int j = 0; j < sample_vec.size(); j++) {
                l.data.fl[j] = sample_vec[j];
            }
            
            float response = knn.find_nearest(&l,1);
//            if (response == 11) cout << "B";
//            else if (response == 10) cout << "A";
//            else cout << response;
            
            if (response == i) trueCount++;
            else falseCount++;
        }
    }
    
    cout << "TRUE: " << trueCount << endl;
    cout << "FALSE: " << falseCount << endl;
}
