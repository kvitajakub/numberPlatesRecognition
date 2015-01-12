#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include "detection.h"
#include "recognition.h"

int main (int argc, char ** argv)
{
    string inputFolder;
    string outputFolder;
    cv::vector<cv::Mat> licencePlates;
    cv::vector<string> file_names;
    string licenceText;
    
    // get agruments from commandline
    for( int i = 1; i < argc; i++){
        if( string(argv[i]) == "-d" && i + 1 < argc){
            inputFolder = argv[++i];
        } else if( string(argv[i]) == "-o" && i + 1 < argc){
            outputFolder = argv[++i];
        } else if( string(argv[i]) == "-h"){
            cout << "Use: " << argv[0] << "  -d inputFolderName -o outputFolderName "<< endl;
            return 0;
        } else {
            cerr << "Use: " << argv[0] << "  -d inputFolderName -o outputFolderName "<< endl;
            return -1;
        }
    }
    
    // check arguments
    if (inputFolder.empty() || outputFolder.empty()) {
        cerr << "Use: " << argv[0] << "  -d inputFolderName -o outputFolderName "<< endl;
        return -1;
    }
    
    // read list of images in directory
    cv::vector<string> images;
    DIR *dir;
    class dirent *ent;
    
    dir = opendir(inputFolder.c_str());
    while ((ent = readdir(dir)) != NULL) {
        string file_name = ent->d_name;
        
        if (file_name[0] == '.')
            continue;
        
        string file_path = inputFolder + "/" + file_name;
        images.push_back(file_path);

        file_names.push_back(file_name);
    }
    closedir(dir);
    
    // go throw all images
    for (int i = 0; i < images.size(); i++) {
        
        ///////// DETECT LICENCE PLATES ////////////
        licencePlates = detect(images[i]);
        cv::Mat image = imread(images[i]);
        
        string imageText;
        // go throw all licence plates and read them
        for (int j = 0; j < licencePlates.size(); j++) {
            licenceText = recognize(licencePlates[j]);
            if (!licenceText.empty()) {
                imageText.append(licenceText);
                imageText.append("  ");
            }
        }
        
        putText(image, imageText, Point(5, 60), FONT_HERSHEY_PLAIN, 2, CV_RGB(0,255,0), 2);
        imwrite(outputFolder + "/" + file_names[i],image);
    }
    
}
