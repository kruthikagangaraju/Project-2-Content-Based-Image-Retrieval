/* Custom Feature - Matching images with image of shoe
* Uses a template to match the foreground of image with foreground of database of images, returning closest matches
Kruthika Gangaraju and Sriram Kodeeswaran */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Buffer to store the name of the file
    char buffer[256];

    // Variables for reading a directory
    DIR* dirp;
    struct dirent* dp;
    vector<double> dist;

    // ImageDetail struct to store the name and value of an image
    struct ImageDetail
    {
        std::string img_name;
        float value;
    };

    // Vector of ImageDetail structs to store the results
    vector<ImageDetail> img_ssd;
    ImageDetail pair;

    // Read the input image
    Mat temp = imread("/Users/illusionsoftruth/Downloads/olympus/pic.0756.jpg", IMREAD_GRAYSCALE);

    // Name of the directory containing the images
    char dirname[] = "/Users/illusionsoftruth/Downloads/olympus";
    printf("Processing directory %s\n", dirname);

    // Open the directory
    dirp = opendir(dirname);
    // Check if the directory could be opened
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // Loop over all the files in the directory
    while ((dp = readdir(dirp)) != NULL) {

        // Check if the file is an image file
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // Build the full path name of the file
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // Read the image file
            Mat image = imread(buffer, IMREAD_GRAYSCALE);

            // Perform template matching between the input image and the current image file
            Mat res;
            matchTemplate(image, temp, res, TM_SQDIFF_NORMED);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

            // Store the result (name and value) in the ImageDetail struct
            pair.value = minVal;
            pair.img_name = buffer;
            // Add the struct to the vector of results
            img_ssd.push_back(pair);
        }
    }
    // Sort the results by value in ascending order

    sort(img_ssd.begin(), img_ssd.end(), [](const ImageDetail& a, const ImageDetail& b)
        {
            return a.value < b.value;
        });
    int N = 11;
    img_ssd.resize(N);


    std::cout << "The top " << N - 1 << " matches are:" << std::endl;

    //    for (auto& n : img_ssd) {
    //        std::cout << n.img_name << ": " << std::fixed << n.value << std::endl;
    //    }
    int j = 0;
    for (auto& n : img_ssd) {
        if (j == 0)
        {
            j++;
            continue;
        }
        std::cout << n.img_name << ": " << std::fixed << n.value << std::endl;
    }
    return 0;
}
