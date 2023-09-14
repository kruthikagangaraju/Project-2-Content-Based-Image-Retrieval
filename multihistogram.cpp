/* Multi Histogram Matching
Kruthika Gangaraju and Sriram Kodeeswaran */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int multi_hist(Mat& img, vector<float>& fvec, int bins)
{
    // Define the size of the histograms as 32
    const int Hsize = 32;
    int dim[3] = { Hsize, Hsize, Hsize };

    // Define two histogram matrices and two histogram vectors for the upper and lower halves of the image
    cv::Mat hist2d1, hist2d2;
    vector<float> histvec1, histvec2;

    // Initialize the histogram matrices with all values set to 0
    hist2d1 = cv::Mat::zeros(3, dim, CV_32S);
    hist2d2 = cv::Mat::zeros(3, dim, CV_32S);

    // Define variables for storing the values of the color channels
    int rx1, ry1, rz1, rx2, ry2, rz2;
    float r1, g1, b1, r2, g2, b2;

    // Define vectors for storing the histogram values
    vector<float> vec1, vec2;

    // Calculate the total number of pixels in half of the image
    float nsum = img.rows * img.cols / 2;

    // Loop through the upper half of the image to calculate the histogram
    for (int i = 0; i < img.rows / 2; i++)
    {
        for (int j = 0; j < img.cols / 2; j++)
        {
            // Get the values of the blue, green, and red channels of the current pixel
            b1 = img.at<cv::Vec3b>(i, j)[0];
            g1 = img.at<cv::Vec3b>(i, j)[1];
            r1 = img.at<cv::Vec3b>(i, j)[2];

            // Calculate the indices in the histogram for each color channel
            rx1 = bins * r1 / (r1 + g1 + b1 + 1);
            ry1 = bins * g1 / (r1 + g1 + b1 + 1);
            rz1 = bins * b1 / (r1 + g1 + b1 + 1);

            // Increment the histogram value at the calculated indices
            hist2d1.at<int>(rx1, ry1, rz1)++;
        }

    }

    // Loop through the bins to normalize the histogram values
    for (int i = 0; i < bins; i++)
    {
        for (int j = 0; j < bins; j++)
        {
            for (int k = 0; k < bins; k++)
            {
                // Normalize the histogram value by dividing by the total number of pixels in half the image
                float normal1 = hist2d1.at<int>(i, j, k) / nsum;
                vec1.push_back(normal1);
            }
        }
    }

    // Loop through the lower half of the image to calculate the histogram


    for (int i = img.rows / 2; i < img.rows; i++) // loop through the lower half of the image rows
        {
            for (int j = img.cols / 2; j < img.cols; j++) // loop through the lower half of the image columns
            {
                b2 = img.at<cv::Vec3b>(i, j)[0]; // retrieve blue channel value
                g2 = img.at<cv::Vec3b>(i, j)[1]; // retrieve green channel value
                r2 = img.at<cv::Vec3b>(i, j)[2]; // retrieve red channel value

                rx2 = bins * r2 / (r2 + g2 + b2+ 1); // calculate bin value for red channel
                ry2 = bins * g2 / (r2 + g2 + b2 + 1); // calculate bin value for green channel
                rz2 = bins * b2 / (r2 + g2 + b2 + 1); // calculate bin value for blue channel

                hist2d2.at<int>(rx2, ry2, rz2)++; // increase the value in the 3D histogram at the corresponding bin
            }

        }

        for (int i = 0; i < bins; i++) // loop through bins
        {
            for (int j = 0; j < bins; j++) // loop through bins
            {
                for (int k = 0; k < bins; k++) // loop through bins
                {
                    float normal2 = hist2d2.at<int>(i, j, k) / nsum; // normalize the histogram value by dividing it with `nsum`
                    vec2.push_back(normal2); // push the normalized value to `vec2`
                }
            }

            fvec.insert(fvec.begin(), vec1.begin(), vec1.end()); // concatenate the two vectors `vec1` and `vec2` into the feature vector `fvec`
            fvec.insert(fvec.end(), vec2.begin(), vec2.end());
        }
        return 0;
    }


int multi_histdist(vector<float>& target_data, vector<string> filenames, vector<vector<float>>& fvec, int N)
{
    // Initialize variables for the histogram intersection
    float intersection;
    double dir1_sum;
    double dir2_sum;
    int bins = 8;
    vector<float> inter1, inter2, inter;
    int n = target_data.size();
    inter.resize(n);
    
    // Initialize variables for the normalized histogram
    float normalized_target;
    float normalized_dir;
    
    // Initialize a vector for storing minimum values
    vector<float> all_min;
    
    // Define a struct for storing image filename and value pairs
    struct ImageDetail
    {
        string img_name;
        float value;
    };
    
    // Initialize vectors for storing upper and lower half histogram intersections
    vector<ImageDetail> img_value1;
    vector<ImageDetail> img_value2;
    vector<ImageDetail> img_value;
    ImageDetail pair1;
    ImageDetail pair2;
    ImageDetail pair;
    
    // Calculate upper half histogram intersection
    for (int i = 0; i < filenames.size(); i++)
    {
        float dir1min_sum = 0;
        for (int j = 0; j < (target_data.size() / 2); j++)
        {
            // Normalize the target data and the feature vector
            normalized_target = target_data[j];
            normalized_dir = fvec[i][j];
            
            // Compute the sum of all normalized minimum values
            dir1min_sum += min(normalized_target, normalized_dir);
        }
        // Store the result in the `inter1` vector
        inter1.push_back(dir1min_sum);
    }
    
    // Calculate lower half histogram intersection
    for (int i = 0; i < filenames.size(); i++)
    {
        float dir2min_sum = 0;
        for (int j = target_data.size() / 2; j < target_data.size(); j++)
        {
            // Normalize the target data and the feature vector
            normalized_target = target_data[j];
            normalized_dir = fvec[i][j];
            
            // Compute the sum of all normalized minimum values
            dir2min_sum += min(normalized_target, normalized_dir);
        }
        // Store the result in the `inter2` vector
        inter2.push_back(dir2min_sum);
    }

    for (int i = 0; i < filenames.size(); i++) {
        // Calculate the final intersection value by taking the average of inter1 and inter2
        inter[i] = (0.5 * inter1[i]) + (0.5 * inter2[i]);

        // Store the image name and the final intersection value in a struct
        pair1.img_name = filenames[i];
        pair1.value = 1 - inter[i];

        // Add the struct to the vector 'img_value'
        img_value.push_back(pair1);
    }

    // Sort the intersection values from the smallest to the largest
    sort(img_value.begin(), img_value.end(), [](const ImageDetail& a, const ImageDetail& b) {
        return a.value < b.value;
    });

    // Resize the vector 'img_value' to contain only the first N elements
    img_value.resize(N);

    // Print the top N matches
    cout << "The top " << N - 1 << " matches are:" << endl;

    int i = 0;
    for (auto& n : img_value) {
        if (i == 0) {
            i++;
            continue;
        }
        // Print the image name and its final intersection value
        std::cout << n.img_name << ": " << std::fixed << n.value << std::endl;
    }

    // Return 0 to indicate successful execution
    return 0;
    }


int main()
{
    // Path to the target image
    string targetpath = "/Users/illusionsoftruth/Downloads/olympus/pic.0274.jpg";
    
    // Read the target image
    Mat targetimg = imread(targetpath);

    // Vector to store the target image's feature vector
    vector<float> targetvec;

    // Directory path to the image files
    char dirname[] = "/Users/illusionsoftruth/Downloads/olympus";

    // Buffer for building the image file's path
    char buffer[256];

    // Pointer to the directory
    DIR* dirp;

    // Structure to store the directory information
    struct dirent* dp;

    // Vector to store the filenames of the images in the directory
    vector<string> dir_filenames;

    // Vector to store the feature vectors of the images in the directory
    vector<vector<float>> dir_fvec;

    // Print the directory path
    printf("Processing directory %s\n", dirname);

    // Open the directory
    dirp = opendir(dirname);

    // Check if the directory could be opened
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // Calculate the feature vector of the target image
    multi_hist(targetimg, targetvec, 8);

    // Loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // Check if the file is an image file
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // Build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // Read the image file
            cv::Mat img;
            img = cv::imread(buffer, cv::IMREAD_COLOR);

            // Check if the image data could be read
            if (img.data == NULL) {
                printf("Unable to read file %s, skipping\n", buffer);
                continue;
            }

            // Vector for image data
            std::vector<float> fvec;

            // Calculate the feature vector of the current image
            multi_hist(img, fvec, 8);

            // Store the filename and feature vector of the current image
            dir_filenames.push_back(dp->d_name);
            dir_fvec.push_back(fvec);
        }
    }

    // Calculate the distance between the target image and the images in the directory
    multi_histdist(targetvec, dir_filenames, dir_fvec, 4);

    return 0;
}
