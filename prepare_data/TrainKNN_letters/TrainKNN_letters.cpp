//
//  TrainKNN_letters.cpp
//  
//
//  Created by Klára Vaňková on 27/12/14.
//
//

#include "Features_ocr.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stack>
#include <dirent.h>
#include <stdio.h>

using namespace std;
using namespace cv;

enum {
    A = 10,
    B
};

int main( int argc, char* argv[])
{
    string inputFolder;
    cv::vector<cv::vector<float> > samples;
    cv::vector<string> responses;
    
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
            samples.push_back(sample_vec);
            responses.push_back(letter);
        }
    }
    
    FileStorage fs("generalsamples.data",FileStorage::WRITE);
    FileStorage fr("generalresponses.data",FileStorage::WRITE);
    
    cv::Mat s(samples.size(), samples.at(0).size(),CV_32FC1);
    for (int j = 0; j < s.rows; j++) {
        for (int k = 0; k < s.cols; k++) {
            s.at<float>(j,k) = samples.at(j).at(k);
        }
    }

    cv::Mat r(responses.size(), 1,CV_32FC1);
    for (int j = 0; j < r.rows; j++) {
        float value = 0;
        
        if (responses[j].compare("A") == 0) value = A;
        else if (responses[j].compare("B") == 0) value = B;
        else value = stoi(responses[j]);
        r.at<float>(j,0) = value;
    }
    
    fs << "samples" << s;
    fr << "responses" << r;

    
    // learn classifier
    CvKNearest knn = CvKNearest();
    knn.train(s, r);
}