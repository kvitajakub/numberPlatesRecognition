//
//  TrainMSER.cpp
//  
//
//  Created by Klára Vaňková on 29/12/14.
//
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


string lettersDir;
string notlettersDir;

/*
 * Train new SVM model on letters and notletters
 */
void train(string lettersDir, string notlettersDir) {
    // paths to images
    cv::vector<string> letters;
    cv::vector<string> notletters;
    
    
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
    
    // training data mat
    int mserCount = letters.size() + notletters.size();
    float trainingData[mserCount][FEATURECOUNT];
    
    // extract features from all letters
    for (int imageNumber = 0; imageNumber < letters.size(); imageNumber++) {
        cv::Mat image = cv::imread(letters[imageNumber],CV_8UC1);
        
        // extract features
        std::vector<float> features = extractFeaturesFromMSER(image);
        
        for (int i = 0; i < FEATURECOUNT; i++) {
            trainingData[imageNumber][i] = features[i];
        }
    }
    
    // extract features from all letters
    for (int imageNumber = 0; imageNumber < notletters.size(); imageNumber++) {
        cv::Mat image = cv::imread(notletters[imageNumber],CV_8UC1);
        
        // extract features
        std::vector<float> features = extractFeaturesFromMSER(image);
        
        for (int i = 0; i < FEATURECOUNT; i++) {
            trainingData[imageNumber + letters.size()][i] = features[i];
        }
    }
    
    
    int countOfMser = letters.size() + notletters.size();
    int countOfLetters = letters.size();
    
    // TRAINING DATA MAT
    cv::Mat trainMat(mserCount,FEATURECOUNT,CV_32FC1);
    
    // create matrix with training data
    for (int i = 0; i < countOfMser; i++) {
        for (int j = 0; j < FEATURECOUNT; j++) {
            trainMat.at<float>(i,j) = trainingData[i][j];
        }
    }
    
    // labels for training data
    float labels[countOfMser];
    for (int i = 0; i < countOfMser; i++) {
        
        labels[i] = (i < countOfLetters) ?  1.0 : -1.0;
    }
    // TRAINING LABELS MAT
    cv::Mat labelsMat = cv::Mat(countOfMser, 1, CV_32FC1, labels);
    
    // Params for SVM
    CvSVMParams params;
    params.svm_type    = CvSVM::NU_SVC;
    params.kernel_type = CvSVM::RBF;
    params.nu = 0.2;
    
    CvSVM SVM;
    
    // Train SVM
    SVM.train_auto(trainMat, labelsMat, cv::Mat(), cv::Mat(), params,10,CvSVM::get_default_grid(CvSVM::C),CvSVM::get_default_grid(CvSVM::GAMMA),CvSVM::get_default_grid(CvSVM::P),CvSVM::get_default_grid(CvSVM::NU),CvSVM::get_default_grid(CvSVM::COEF),CvSVM::get_default_grid(CvSVM::DEGREE),true);
    
    SVM.save("classifier_mser.xml");
}

int main(int argc, char *argv[]) {
    string lettersDir;
    string notlettersDir;
    
    // zpracovani parametru prikazove radky
    for( int i = 1; i < argc; i++){
        if( string(argv[i]) == "-l" && i + 1 < argc){
            lettersDir = argv[++i];
        } else if( string(argv[i]) == "-n" && i + 1 < argc){
            notlettersDir = argv[++i];
        }else if( string(argv[i]) == "-h"){
            cout << "Use: " << argv[0] << "  -d inputFolder "<< endl;
            return 0;
        } else {
            cerr << "Use: " << argv[0] << "  -d inputFolder "<< endl;
            return -1;
        }
    }
    
    // kontrola zadani parametru
    if( lettersDir.empty() || notlettersDir.empty()) {
        cerr << "Use: " << argv[0] << "  -d inputFolder "<< endl;
        return -1;
    }
    
    // train SVM model
    train(lettersDir, notlettersDir);
}

