#include "recognition.h"
#include "Features_ocr.h"
#include "Features_char.h"

#define FEATURECOUNT 9

// SVM model for letters/notletters
CvSVM SVM_mser;

// missing letters
string miss = "";


template <typename T>
string NumberToString ( T Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

CvRect makeBoundingBox(std::vector<cv::Point> blob){

    cv::Point minpoint(blob[0]);
    cv::Point maxpoint(blob[0]);

    for(int i=1;i<blob.size();i++){
        if(minpoint.x > blob[i].x)
            minpoint.x = blob[i].x;
        if(minpoint.y > blob[i].y)
            minpoint.y = blob[i].y;
        if(maxpoint.x < blob[i].x)
            maxpoint.x = blob[i].x;
        if(maxpoint.y < blob[i].y)
            maxpoint.y = blob[i].y;
    }

    // maxpoint.x++;
    // maxpoint.y++;

    return cv::Rect(minpoint,maxpoint);

}


// find black blobs in image - possible letters
void findBlobs(bool saveSmall, const cv::Mat &binary, std::vector < std::vector<cv::Point> > &blobs, cv::vector<CvRect> &boundingBoxes)
{
    blobs.clear();
    
    // Fill the label_image with the blobs
    // 255  - foreground
    int label_count = 2;
    
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            
            if(row[x] != 255) {
                continue;
            }
            
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 8);

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
            
            
            // check if it isnt too small or too big
            CvRect boundRect = makeBoundingBox(blob);

            // dont save too small or too big
            if (!saveSmall && ((boundRect.height < binary.rows / 3) || (boundRect.width > binary.cols / 5) || ((binary.cols - boundRect.height) < 5))) {
                continue;
            }
            
            // save it
            blobs.push_back(blob);
            boundingBoxes.push_back(boundRect);
            
            label_count++;
        }
    }
}

/**
 * Decide if image is letter or not.
 */
bool isLetter(cv::Mat image) {
    // extract features from MSER
    float sampleData[1][FEATURECOUNT];
    std::vector<float> features;
    features = extractFeaturesFromMSER(image);
    
    for (int i = 0; i < FEATURECOUNT; i++) {
        sampleData[0][i] = features[i];
    }
    
    // create sample Mat for SVM
    cv::Mat sampleMat = cv::Mat(1,FEATURECOUNT,CV_32FC1,sampleData);
    
    return SVM_mser.predict(sampleMat);
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

/**
 * Find first black pixel from the top in image in specified column.
 */
int getTopBlackPixelInCol(cv::Mat image, int col) {
    int i = 0;
    while ((image.at<char>(i,col) == 0) && (i < image.rows)) {
        i++;
    }
    if ((i == image.rows) && (image.at<char>(image.rows-1,col) == 0)) {
        return 0;
    }
    return i;
}

/**
 * Find first black pixel bottom the top in image in specified column.
 */
int getBottomBlackPixelInCol(cv::Mat image, int col) {
    int i = image.rows;
    while ((image.at<char>(i,col) == 0) && (i >= 0)) {
        i--;
    }
    if (i == -1) {
        return image.rows;
    }
    
    return i;
}

/**
 * Find where to cut two connected letters.
 */
int findCut(cv::Mat twoLetters) {
    int min = twoLetters.rows;
    int index = 0;
    for (int i = 0; i < twoLetters.cols; i++) {
        int top = getTopBlackPixelInCol(twoLetters,i);
        int bottom = getBottomBlackPixelInCol(twoLetters,i);
        int diff = bottom - top;
        if ((diff != 0) && (diff < min)) {
            min = diff;
            index = i;
        }
    }
    
    return index;
}

/**
 * Remove bad blobs which cannot be letters or numbers.
 */
void removeBad(std::vector < std::vector<cv::Point> > &blobs, cv::vector<CvRect> &boundingBoxes, cv::Mat licencePlate) {
    
    // find mean values
    int top = 0;
    int width = 0;
    int height = 0;
    
    for (int i = 0; i < boundingBoxes.size(); i++) {
        top += boundingBoxes[i].y;
        width += boundingBoxes[i].width;
        height += boundingBoxes[i].height;
    }
    
    int meanTop = top / boundingBoxes.size();
    int meanWidth = width / boundingBoxes.size();
    int meanHeight = height / boundingBoxes.size();
    
    // go throw all blobs
    for (int i = 0; (i < blobs.size() && blobs.size() > 0); i++) {
        
        // remove letters not in line and with different height
        if ((abs(boundingBoxes[i].y - meanTop) > 5) ||
            (abs(boundingBoxes[i].height - meanHeight) > 5)){
            // erase old big one
            blobs.erase(blobs.begin() + i);
            boundingBoxes.erase(boundingBoxes.begin() + i);
            i--;
            continue;
        }
        
        // try cut too big ones as two letters
        if ((boundingBoxes[i].width - meanWidth) > 5) {
            cv::Mat twoLetters = drawMser(blobs[i],licencePlate);
            
            int cut = findCut(twoLetters);
            
            // Msers
            cv::vector<cv::vector<cv::Point> > twoLettersBlobs;
            // Msers bounding boxes
            cv::vector<CvRect> boxes;
            
            // mask surroundings
            cv::Mat licenceCopy;
            licencePlate.copyTo(licenceCopy);
            
            for (int k = 0; k < licenceCopy.rows; k++) {
                for (int l = 0; l < licenceCopy.cols; l++) {
                    if ( (k < boundingBoxes[i].y)
                        || (k > boundingBoxes[i].y + boundingBoxes[i].height)
                        || (l < boundingBoxes[i].x)
                        || (l > boundingBoxes[i].x + boundingBoxes[i].width))
                    licenceCopy.at<char>(k,l) = 0;
                }
            }
            
            cut += boundingBoxes[i].x;
            // black cutting column
            for (int j = 0; j < licenceCopy.rows; j++) { licenceCopy.at<char>(j,cut) = licenceCopy.at<char>(0,0); }
            
            // erase old big one
            blobs.erase(blobs.begin() + i);
            boundingBoxes.erase(boundingBoxes.begin() + i);
            i--;
            
            findBlobs(true, licenceCopy,twoLettersBlobs,boxes);
            if (twoLettersBlobs.size() < 2) {
                continue;
            }
            
            cv::Mat first = drawMser(twoLettersBlobs[0],licenceCopy);
            cv::Mat second = drawMser(twoLettersBlobs[1],licenceCopy);
            
            // insert new ones
            if (isLetter(first)) {
                blobs.push_back(twoLettersBlobs[0]);
                boundingBoxes.push_back(makeBoundingBox(twoLettersBlobs[0]));
            }
            if (isLetter(second)) {
                blobs.push_back(twoLettersBlobs[1]);
                boundingBoxes.push_back(makeBoundingBox(twoLettersBlobs[1]));
            }
        }
    }
    
    // if more than 7 blobs - try to remove not letters
    if (blobs.size() > 7) {
        // go throw all blobs
        for (int i = 0; (i < blobs.size() && blobs.size() > 0); i++) {
            cv::Mat mser = drawMser(blobs[i],licencePlate);
            if (!isLetter(mser)) {
                blobs.erase(blobs.begin() + i);
                boundingBoxes.erase(boundingBoxes.begin() + i);
                i--;
            }
        }
    }
}

/**
 * Return the most right letter index.
 */
int findMostRight(cv::vector<CvRect> &boundingBoxes) {
    int right = 0;
    for (int i = 1; i < boundingBoxes.size(); i++) {
        if (boundingBoxes[i].x > boundingBoxes[right].x) {
            right = i;
        }
    }
    
    return right;
}

/**
 * Return the most left letter index.
 */
int findMostLeft(cv::vector<CvRect> &boundingBoxes) {
    int left = 0;
    for (int i = 1; i < boundingBoxes.size(); i++) {
        if (boundingBoxes[i].x < boundingBoxes[left].x) {
            left = i;
        }
    }
    
    return left;
}

/**
 * Locate and add missing letter.
 */
void findMissing(const cv::Mat &binary, std::vector < std::vector<cv::Point> > &blobs, cv::vector<CvRect> &boundingBoxes)
{
    
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    
    std::vector<int> order;
    int left;
    
    for(int i=0; i<blobs.size(); i++){
        left = findMostRight(boundingBoxes);
        for(int j=0; j<blobs.size();j++){
            if(order.size() == 0){
                left=findMostLeft(boundingBoxes);
                break;
            }else{
                if(boundingBoxes[order.back()].x<boundingBoxes[j].x
                   && boundingBoxes[j].x < boundingBoxes[left].x){
                    
                    left = j;
                    
                }
            }
        }
        order.push_back(left);
        // std::cout<<boundingBoxes[order.back()].width << std::endl;
    }
    
    int meanWidth=0;
    for(int i=0; i<order.size(); i++){
        meanWidth += boundingBoxes[order[i]].width;
    }
    meanWidth /= order.size();
    // std::cout<<"meanWidth = " <<meanWidth <<std::endl;
    
    
    int meanHeight=0;
    for(int i=0; i<order.size(); i++){
        meanHeight += boundingBoxes[order[i]].height;
        // imwrite(NumberToString(i)+" aaa.png",drawMser(blobs[order[i]],binary));
    }
    meanHeight /= order.size();
    // std::cout<<"meanHeight = " <<meanHeight <<std::endl;
    
    
    std::vector<int> spaces;
    
    for(int i=1;i<order.size();i++){
        
        spaces.push_back(boundingBoxes[order[i]].x
                         -boundingBoxes[order[i-1]].x
                         -boundingBoxes[order[i-1]].width
                         -1);
        // std::cout<< spaces.back() <<std::endl;
    }
    
    if(blobs.size()==7){
        return;
    }
    else if(blobs.size()==6){
        //chybi jedna
        
        cv::Rect missing;
        
        
        //velikosti mezery
        //1 mala
        //2 velka
        //3 fakt welka
        
        // 1 2 1 1 1 chybi 1
        // 2 2 1 1 1 chybi 2
        // 1 3 1 1 1 chybi 3
        // 1 1 3 1 1 chybi 4
        // 1 1 2 2 1 chybi 5
        // 1 1 2 1 2 chybi 6
        // 1 1 2 1 1 chybi 7
        
        if(spaces[0]>meanWidth){
            //chybi 2
//            std::cout <<"Chybi 2" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[0]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[0]].x
            +boundingBoxes[order[0]].width
            +1;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else if(spaces[1] > meanWidth*5){
            //chybi 3
//            std::cout <<"Chybi 3" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[1]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[1]].x
            +boundingBoxes[order[1]].width
            +1;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else if(spaces[1]>meanWidth){
            //chybi 1
//            std::cout <<"Chybi 1" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[0]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[0]].x
            -boundingBoxes[order[0]].width
            -2;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else if(spaces[2]>meanWidth*3){
            //chybi 4
//            std::cout <<"Chybi 4" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[3]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[3]].x
            -boundingBoxes[order[3]].width
            -3;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else if(spaces[3]>meanWidth){
            //chybi 5
//            std::cout <<"Chybi 5" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[3]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[3]].x
            +boundingBoxes[order[3]].width
            +1;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else if(spaces[4]>meanWidth){
            //chybi 6
//            std::cout <<"Chybi 6" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[4]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[4]].x
            +boundingBoxes[order[4]].width
            +1;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        else{
            //chybi 7
//            std::cout <<"Chybi 7" <<std::endl;
            
            missing.width = meanWidth+3;
            missing.height = meanHeight+3;
            missing.y = boundingBoxes[order[5]].y-1;
            missing.y = missing.y>=0?missing.y : 0;
            missing.x = boundingBoxes[order[5]].x
            +boundingBoxes[order[5]].width
            +1;
            missing.x = missing.x>=0?missing.x : 0;
            
        }
        
//        std::cout<< missing.x <<"   "<<missing.y<<std::endl;
//        std::cout<< missing.width <<"   "<<missing.height<<std::endl;
        
        // mask surroundings
        cv::Mat licenceCopy;
        binary.copyTo(licenceCopy);
        
        for (int k = 0; k < licenceCopy.rows; k++) {
            for (int l = 0; l < licenceCopy.cols; l++) {
                if ( (k < missing.y)
                    || (k > missing.y + missing.height)
                    || (l < missing.x)
                    || (l > missing.x + missing.width))
                    licenceCopy.at<char>(k,l) = 0;
            }
        }
        
        // Msers
        cv::vector<cv::vector<cv::Point> > letterBlob;
        // Msers bounding boxes
        cv::vector<CvRect> box;
        
        findBlobs(true, licenceCopy,letterBlob,box);
        
        if (letterBlob.size() == 0) {
            return;
        }
        // get the bigger one
        int selected = 0;
        int area = box[0].height * box[0].width;
        for (int i = 1; i < letterBlob.size(); i++) {
            int tmp = box[i].height * box[i].width;
            if (area < tmp) {
                area = tmp;
                selected = i;
            }
        }
        
        blobs.push_back(letterBlob[selected]);
        boundingBoxes.push_back(makeBoundingBox(letterBlob[selected]));
    }
    else if(blobs.size()==5){
        //TODO
        //chybi dve
        
        //velikosti mezer a co znamenaji
        // 4 1 1 1 chybi 2 a 3
        // 3 1 1 1 chybi 1 a 3
        // 2 3 1 1 chybi 2 a 4
        // 2 2 2 1 chybi 2 a 5
        // 2 2 1 2 chybi 2 a 6
        // 2 2 1 1 chybi 2 a 7
        // 2 1 1 1 chybi 1 a 2
        // 1 4 1 1 chybi 3 a 4
        // 1 3 2 1 chybi 3 a 5
        // 1 3 1 2 chybi 3 a 6
        // 1 3 1 1 chybi 3 a 7
        // 1 3 1 1 chybi 1 a 4
        // 1 2 2 1 chybi 1 a 5
        // 1 2 1 2 chybi 1 a 6
        // 1 2 1 1 chybi 1 a 7
        // 1 1 4 1 chybi 4 a 5
        // 1 1 3 2 chybi 4 a 6
        // 1 1 3 1 chybi 4 a 7
        // 1 1 2 3 chybi 5 a 6
        // 1 1 2 2 chybi 5 a 7
        // 1 1 2 1 chybi 6 a 7
        
        if(spaces[0]>meanWidth*3){
//            std::cout <<"Chybi 2 a 3" <<std::endl;
            miss = "chybi 2 a 3";
            
        }
        else if(spaces[0] > meanWidth*2){
//            std::cout <<"Chybi 1 a 3" <<std::endl;
            miss = "chybi 1 a 3";
        }
        else if(spaces[0] > meanWidth
                && spaces[1] > meanWidth*2){
//            std::cout <<"Chybi 2 a 4" <<std::endl;
            miss = "chybi 2 a 4";
            
        }
        else if(spaces[0] > meanWidth
                && spaces[1] > meanWidth
                && spaces[2] > meanWidth){
//            std::cout <<"Chybi 2 a 5" <<std::endl;
            miss = "chybi 2 a 5";
            
        }
        else if(spaces[0] > meanWidth
                && spaces[1] > meanWidth
                && spaces[3] > meanWidth){
//            std::cout <<"Chybi 2 a 6" <<std::endl;
            miss = "chybi 2 a 6";
            
        }
        else if(spaces[0] > meanWidth
                && spaces[1] > meanWidth){
//            std::cout <<"Chybi 2 a 7" <<std::endl;
            miss = "chybi 2 a 7";
            
        }
        else if(spaces[0] > meanWidth){
//            std::cout <<"Chybi 1 a 2" <<std::endl;
            miss = "chybi 1 a 2";
        }
        else if(spaces[1] > meanWidth*3){
//            std::cout <<"Chybi 3 a 4" <<std::endl;
            miss = "chybi 3 a 4";
            
        }
        else if(spaces[1] > meanWidth*2
                && spaces[2] > meanWidth){
//            std::cout <<"Chybi 3 a 5" <<std::endl;
            miss = "chybi 3 a 5";
            
        }
        else if(spaces[1] > meanWidth*2
                && spaces[3] > meanWidth){
//            std::cout <<"Chybi 3 a 6" <<std::endl;
            miss = "chybi 3 a 6";
            
        }
        else if(spaces[1] > meanWidth*2){
//            std::cout <<"Chybi 3 a 1 nebo! 1 a 4" <<std::endl;
            miss = "chybi 3 a 1 nebo! 1 a 4";
            
        }
        else if(spaces[1] > meanWidth
                && spaces[2] > meanWidth){
//            std::cout <<"Chybi 1 a 5" <<std::endl;
            miss = "chybi 1 a 5";
            
        }
        else if(spaces[1] > meanWidth
                && spaces[3] > meanWidth){
//            std::cout <<"Chybi 1 a 6" <<std::endl;
            miss = "chybi 1 a 6";
            
        }
        else if(spaces[1] > meanWidth){
//            std::cout <<"Chybi 1 a 7" <<std::endl;
            miss = "chybi 1 a 7";
            
        }
        else if(spaces[2] > meanWidth*3){
//            std::cout <<"Chybi 4 a 5" <<std::endl;
            miss = "chybi 4 a 5";
            
        }
        else if(spaces[2] > meanWidth*2
                && spaces[3] > meanWidth){
//            std::cout <<"Chybi 4 a 6" <<std::endl;
            miss = "chybi 4 a 6";
            
        }
        else if(spaces[2] > meanWidth*2){
//            std::cout <<"Chybi 4 a 7" <<std::endl;
            miss = "chybi 4 a 7";
            
        }
        else if(spaces[2] > meanWidth
                && spaces[3] > meanWidth*2){
//            std::cout <<"Chybi 5 a 6" <<std::endl;
            miss = "chybi 5 a 6";
            
        }
        else if(spaces[2] > meanWidth
                && spaces[3] > meanWidth){
//            std::cout <<"Chybi 5 a 7" <<std::endl;
            miss = "chybi 5 a 7";
            
        }
        else if(spaces[2] > meanWidth){
//            std::cout <<"Chybi 6 a 7" <<std::endl;
            miss = "chybi 6 a 7";
            
        }
    }
}


string recognize(cv::Mat licencePlate) {
    // final text
    string text;
    // Msers
    cv::vector<cv::vector<cv::Point> > msers;
    // Msers bounding boxes
    cv::vector<CvRect> boundingBoxes;

    /////////// PREPARE IMAGE //////////////
    // need one channel img
    if (licencePlate.channels() != 1)
        cv::cvtColor(licencePlate, licencePlate, CV_RGB2GRAY);
    
    string index = NumberToString(rand());
    threshold( licencePlate, licencePlate, 0, 255,CV_THRESH_OTSU);
    // invert
    for (int i = 0; i < licencePlate.rows; i++) {
        for (int j = 0; j < licencePlate.cols; j++) {
            if (licencePlate.at<char>(i,j) == 0) licencePlate.at<char>(i,j) = 255;
            else licencePlate.at<char>(i,j) = 0;
        }
    }
    
    
    /////////// FIND POSSIBLE LETTERS //////////////
    
    // get all black areas
    findBlobs(false, licencePlate, msers, boundingBoxes);
    
    // if no blob in image
    if (msers.size() == 0) {
        return text;
    }
    
    // load SVM model
    SVM_mser.load("mser_model/classifier_mser.xml");
    
    // remove bad black areas
    removeBad(msers, boundingBoxes, licencePlate);
    
    // if less than 5 blobs - it probably isnt licence plate
    if ((msers.size() < 5) || (msers.size() > 9)) {
        return text;
    }


    findMissing(licencePlate, msers, boundingBoxes);


    // DRAW BLOBS
    cv::Mat output = cv::Mat::zeros(licencePlate.size(), CV_8UC3);
    // Randomy color the blobs
    for(size_t i=0; i < msers.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
        
        for(size_t j=0; j < msers[i].size(); j++) {
            int x = msers[i][j].x;
            int y = msers[i][j].y;
            
            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }
    
    /////////// LOAD KNN SAMPLES AND RESPONSES //////////////
    cv::Mat samples;
    cv::Mat responses;
    FileStorage samplesFile("ocr_models/generalsamples.data",FileStorage::READ);
    FileStorage responsesFile("ocr_models/generalresponses.data",FileStorage::READ);
    
    samplesFile["samples"] >> samples;
    responsesFile["responses"] >> responses;
    
    /////////// LEARN KNN //////////////
    CvKNearest knn = CvKNearest();
    knn.train(samples, responses);

    //cout << index << ":" ;
    for (int i = msers.size(); i >0 ; i--) {
        int left = findMostLeft(boundingBoxes);
        cv::Mat mserImg = drawMser(msers[left],licencePlate);
        
        msers.erase(msers.begin() + left);
        boundingBoxes.erase(boundingBoxes.begin() + left);
       
        
        cv::vector<float> sample_vec = extractOCRFeatures(mserImg);
        
        float _l[sample_vec.size()];
        CvMat l = cvMat( 1, sample_vec.size(), CV_32FC1, _l );
        for (int j = 0; j < sample_vec.size(); j++) {
            l.data.fl[j] = sample_vec[j];
        }
        
        float response = knn.find_nearest(&l,1);
        if (response == 11) text.append("B");
        else if (response == 10) text.append("A");
        else text.append(NumberToString(response));
    }
    text.append(" ");
    text.append(miss);
    miss = "";
    
    return text;
}