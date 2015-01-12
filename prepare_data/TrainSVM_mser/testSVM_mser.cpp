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
#include "Features_char.h"

#define FEATURECOUNT 9

using namespace std;
using namespace cv;


int main() {
    // load SVM model
    CvSVM SVM;
    SVM.load("classifier_mser.xml");
    // SVM decide if MSER is letter or not
    
    string lettersDir = "ref_letters";
    string notlettersDir = "ref_notletters";
    
    // paths to images
    cv::vector<string> letters;
    cv::vector<string> notletters;
    
    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    
    // get paths to images with letters
    DIR *dir;
    class dirent *ent;
    
    dir = opendir(lettersDir.c_str());
    while ((ent = readdir(dir)) != NULL) {
        string file_name = ent->d_name;
        
        if (file_name[0] == '.')
            continue;
        
        string file_path = lettersDir + "/" + file_name;
        letters.push_back(file_path);
    }
    closedir(dir);
    
    // get paths to images with letters
    dir = opendir(notlettersDir.c_str());
    while ((ent = readdir(dir)) != NULL) {
        string file_name = ent->d_name;
        
        if (file_name[0] == '.')
            continue;
        
        string file_path = notlettersDir + "/" + file_name;
        notletters.push_back(file_path);
    }
    closedir(dir);
    
    // CHECK LETTERS
    for (int i = 0; i < letters.size(); i++) {
        cv::Mat image = cv::imread(letters[i],CV_8UC1);
        
        // extract features from MSER
        float sampleData[1][FEATURECOUNT];
        std::vector<float> features;
        features = extractFeaturesFromMSER(image);

        for (int i = 0; i < FEATURECOUNT; i++) {
            sampleData[0][i] = features[i];
        }
        
        // create sample Mat for SVM
        cv::Mat sampleMat = cv::Mat(1,FEATURECOUNT,CV_32FC1,sampleData);
        
        float response = SVM.predict(sampleMat);

        // result
        if (response == 1) truePositive++;
        else falseNegative++;
    }

    // CHECK LETTERS
    for (int i = 0; i < notletters.size(); i++) {
        cv::Mat image = cv::imread(notletters[i],CV_8UC1);
        
        // extract features from MSER
        float sampleData[1][FEATURECOUNT];
        std::vector<float> features;
        features = extractFeaturesFromMSER(image);
        
        for (int i = 0; i < FEATURECOUNT; i++) {
            sampleData[0][i] = features[i];
        }
        
        // create sample Mat for SVM
        cv::Mat sampleMat = cv::Mat(1,FEATURECOUNT,CV_32FC1,sampleData);
        
        float response = SVM.predict(sampleMat);
        
        // result
        if (response == 1) falsePositive++;
        else trueNegative++;
    }
    
    cout << "TRUE POSITIVE: " << truePositive << endl;
    cout << "FALSE POSITIVE: " << falsePositive << endl;
    cout << "TRUE NEGATIVE: " << trueNegative << endl;
    cout << "FALSE NEGATIVE: " << falseNegative << endl;
    
}
