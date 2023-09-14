/*Kruthika Gangaraju and Sriram Kodeeswaran */

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

// Namespaces being used in the code
using namespace std;
using namespace cv;

// Function to calculate the 2D histogram of an image
int histogram(cv::Mat& img, std::vector<float>& fvec, int num_bins)
{
    // Constants for the size of the histogram and the dimension
    const int Hsize = 32;
    int dim[2] = { Hsize, Hsize };

    // Initializing the histogram matrix
    cv::Mat hist2d;
    hist2d = cv::Mat::zeros(2, dim, CV_32S);
    
    // Loop variables
    int i, j, rx, ry;
    float r, g, b;

    // Looping through the rows and columns of the input image
    for (i = 0; i < img.rows; i++) {
        for (j = 0; j < img.cols; j++) {
            // Get the blue, green, and red values of the current pixel
            b = img.at<cv::Vec3b>(i, j)[0];
            g = img.at<cv::Vec3b>(i, j)[1];
            r = img.at<cv::Vec3b>(i, j)[2];

            // Calculate the bin index for the red and green values
            rx = num_bins * r / (r + b + g + 1);
            ry = num_bins * g / (r + b + g + 1);

            // Increment the histogram value for the corresponding bin
            hist2d.at<int>(rx, ry)++;
        }

    }

    // Push the histogram values to the output vector
    for (i = 0; i < hist2d.rows; i++) {
        for (j = 0; j < hist2d.cols; j++) {
            fvec.push_back(hist2d.at<int>(i, j));
        }
    }

    // Return success status
    return 0;
}

int histx(std::vector<float>& target_data, vector<string> dir_filename, std::vector<std::vector<float>>& dir_fvec, int N) {
    // Define variables for the sum of the target image data and the intersection value
    float intersection;
    double target_sum = 0;
    double dir_sum = 0;

    // Define variables for the normalized histograms
    float normalized_target;
    float normalized_dir;

    // Define a struct to store image filename and value pairs
    struct ImageStruct
    {
        string img_name;
        float value;
    };
    std::vector<ImageStruct> img_value;
    ImageStruct pair;

    // Calculate the sum of the target image data
    target_sum = accumulate(target_data.begin(), target_data.end(), 0);

    // Loop through each directory image to calculate the sum, find the minimum values,
    // and populate the values in a vector
    for (int i = 0; i < dir_filename.size(); i++) {
        // Calculate the sum of the directory image data
        dir_sum = accumulate(dir_fvec[i].begin(), dir_fvec[i].end(), 0);

        // Define a variable to keep track of the sum of all minimum values
        float min_sum = 0;

        // Loop through each target data to normalize the values and compute the minimum
        for (int j = 0; j < target_data.size(); j++) {
            // Normalize the target and directory image values
            normalized_target = target_data[j] / target_sum;
            normalized_dir = dir_fvec[i][j] / dir_sum;

            // Calculate the sum of all minimum values
            min_sum += min(normalized_target, normalized_dir);
        }

        // Calculate the histogram intersection by subtracting the minimum sum from 1
        intersection = abs(1 - min_sum);

        // Store the image filename and intersection value as a pair in a vector
        pair.img_name = dir_filename[i];
        pair.value = intersection;
        img_value.push_back(pair);
    }

    // Sort the image-value pairs in ascending order of the values
    sort(img_value.begin(), img_value.end(), [](const ImageStruct& a, const ImageStruct& b) {
        return a.value < b.value;
        });

    // Keep only the first N-1 image-value pairs
    img_value.resize(N);

    // Print the top N-1 matches
    std::cout << "The top " << N-1 << " matches are:" << std::endl;
    int i=0;
    for (auto& n : img_value) {
        if(i==0)
        {i++;
            continue;}
        std::cout << n.img_name  << ": " << std::fixed << n.value << std::endl;
    }

    // Return 0 indicating success
    return 0;
}

int main()
{
    // Define the path to the target image
    string targetpath = "/Users/illusionsoftruth/Downloads/olympus/pic.0164.jpg";

    // Load the target image into a matrix
    Mat targetimg = imread(targetpath);

    // Create a vector to hold the histogram of the target image
    vector<float> targetvec;

    // Define the directory path
    char dirname[] = "/Users/illusionsoftruth/Downloads/olympus";
    char buffer[256];

    // Create a pointer to a directory object
    DIR* dirp;

    // Create a pointer to a directory entry object
    struct dirent* dp;

    // Create a vector to store the filenames of all images in the directory
    vector<string> dir_filenames;

    // Create a vector to store the histograms of all images in the directory
    vector<vector<float>> dir_fvec;

    // Print the directory path
    printf("Processing directory %s\n", dirname);

    // Open the directory
    dirp = opendir(dirname);

    // Check if the directory can be opened
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // Compute the histogram of the target image
    histogram(targetimg, targetvec,16);

    // Loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // Check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // Build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // Load the image into a matrix object
            cv::Mat img;
            img = cv::imread(buffer, cv::IMREAD_COLOR);

            // Check if the image data can be read
            if (img.data == NULL) {
                printf("Unable to read file %s, skipping\n", buffer);
                continue;
            }

            // Create a vector to hold the histogram of the current image
            std::vector<float> fvec;

            // Compute the histogram of the current image
            histogram(img, fvec,16);

            // If the directory name is not ".", "..", or "pic.0164.jpg",
            // add the filename and histogram vector to the corresponding vectors
            if (dirname != "." && dirname != ".." && dirname != "pic.0164.jpg" ) {
                dir_filenames.push_back(dp->d_name);
                dir_fvec.push_back(fvec);
            }
        }
    }
    // Compare the histograms of the target image and the other images in the directory
    histx(targetvec, dir_filenames, dir_fvec, 4);
    return 0;
}
