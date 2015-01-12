#include "Features_ocr.h"
//
//int inMSER = -1;
//int notInMSER = 0;

void downsample(cv::Mat *mser) {
    pyrDown(*mser, *mser, Size( (*mser).cols/2, (*mser).rows/2));
    pyrDown(*mser, *mser, Size( (*mser).cols/2, (*mser).rows/2));
}

std::vector<float> distanceFromLeft(cv::Mat letter) {
    std::vector<float> distances;
    
    for (int i = 0; i < letter.rows; i++) {
        float d = 0;
        int j = 0;
        
        while ((letter.at<char>(i,j) == 0) && (j < letter.cols)) {
            j++;
            d++;
        }
        
        if (i%2 == 1) {
            distances.push_back(d/2);
            d = 0;
        }
    }
    
    return distances;
}

std::vector<float> distanceFromRight(cv::Mat letter) {
    std::vector<float> distances;
    
    for (int i = letter.rows - 1; i >= 0; i--) {
        float d = 0;
        int j = letter.cols - 1;
        
        while ((letter.at<char>(i,j) == 0) && (j >= 0)) {
            j--;
            d++;
        }
        
        if (i%2 == 1) {
            distances.push_back(d/2);
            d = 0;
        }
    }
    return distances;
}

std::vector<float> countChanges(cv::Mat letter) {
    std::vector<float> changes;
    
    for (int i = 0; i < letter.rows; i++) {
        float d = 0;
        
        for (int j = 1; j < letter.cols; j++) {
            if (letter.at<char>(i,j) != letter.at<char>(i,j-1)) {
                d++;
            }
        }
        
        if (i%2 == 1) {
            changes.push_back(d/2);
            d = 0;
        }
    }
    return changes;
}

std::vector<float> extractOCRFeatures(cv::Mat letter) {
    std::vector<float> features;
    resize(letter,letter,Size(16,28));
    
    // eight bitmaps
    cv::Mat right = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat left = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat up = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat down = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat upRight = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat upLeft = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat downRight = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    cv::Mat downLeft = Mat::zeros(letter.rows, letter.cols, CV_8UC1);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat dst = Mat::zeros(letter.rows, letter.cols, CV_8UC3);
    
    findContours( letter, contours, hierarchy,
                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    
    for (int i = 0; i < contours.size(); i++) {
        for (int j = 1; j < contours[i].size(); j++) {
            
            //cout << contours[i][j] << ",";
            
            // DOWN
            if (contours[i][j-1].y < contours[i][j].y) {
                //cout << "down";
                if (contours[i][j-1].x == contours[i][j].x) {
                    cv::line(down, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << ",";
                } else if (contours[i][j-1].x > contours[i][j].x) {
                    cv::line(downLeft, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " left,";
                } else {
                    cv::line(downRight, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " right,";
                }
            }
            // LEFT or RIGHT
            else if (contours[i][j-1].y == contours[i][j].y) {
                if (contours[i][j-1].x > contours[i][j].x) {
                    cv::line(left, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " left,";
                } else {
                    cv::line(right, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " right,";
                }
            }
            // UP
            else {
                //cout << "up";
                if (contours[i][j-1].x == contours[i][j].x) {
                    cv::line(up, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << ",";
                } else if (contours[i][j-1].x > contours[i][j].x) {
                    cv::line(upLeft, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " left,";
                } else {
                    cv::line(upRight, contours[i][j-1], contours[i][j], Scalar(255,255,255));
                    //cout << " right,";
                }
            }
        }
        //cout << endl;
    }
    //cout << endl;
    
    downsample(&down);
    downsample(&up);
    downsample(&left);
    downsample(&right);
    downsample(&downLeft);
    downsample(&downRight);
    downsample(&upLeft);
    downsample(&upRight);
    
    cv::Mat sample_mat;
    cv::vector<float> sample_vec;
    
    sample_mat = down.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = up.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = left.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = right.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = downLeft.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = downRight.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = upLeft.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_mat = upRight.reshape(0,1);
    sample_mat.row(0).copyTo(sample_vec);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_vec = distanceFromLeft(letter);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
   
    sample_vec = distanceFromRight(letter);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    sample_vec = countChanges(letter);
    features.insert(features.end(), sample_vec.begin(), sample_vec.end());
    
    distanceFromLeft(letter);
    
    
    return features;
}