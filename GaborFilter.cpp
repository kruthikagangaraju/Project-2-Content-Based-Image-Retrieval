/* Gabor Filter Histogram
* Uses a Gabor Filter to extract textures of images which is combined with color fvec to generate histogram intersection
Kruthika Gangaraju and Sriram Kodeeswaran*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>


using namespace std;
using namespace cv;

Mat gaborFilter(Size ksize, double sigma, double theta, double lambda, double gamma)
{
    // Create a kernel of specified size and type CV_32F (32-bit floating-point)
    Mat kernel(ksize, CV_32F);

    // Define sigma_x and sigma_y, where sigma_x = sigma and sigma_y = sigma / gamma
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;

    // Find the center of the kernel
    int x_c = ksize.width / 2;
    int y_c = ksize.height / 2;

    // Loop over each pixel in the kernel
    for (int x = 0; x < ksize.width; x++)
    {
        for (int y = 0; y < ksize.height; y++)
        {
            // Rotate the pixel coordinate based on theta
            double x_ = (x - x_c) * cos(theta) + (y - y_c) * sin(theta);
            double y_ = -(x - x_c) * sin(theta) + (y - y_c) * cos(theta);

            // Calculate the Gaussian function based on the rotated pixel coordinate
            double g = exp(-(x_ * x_) / (2 * sigma_x * sigma_x) - (y_ * y_) / (2 * sigma_y * sigma_y));

            // Calculate the harmonic function based on the rotated pixel coordinate
            double h = cos(2 * M_PI * x_ / lambda);

            // Combine the Gaussian and harmonic functions to generate the Gabor filter kernel
            kernel.at<float>(y, x) = (float)(g * h);
        }
    }

    // Return the generated Gabor filter kernel
    return kernel;
}


// Function to extract texture and color features from an image
int texturecolor(Mat& img, vector<float>& fvec, int bins)
{
    /* TEXTURE */
    // Apply Gabor
    Size ksize(101, 101); // kernel size for the Gabor filter
    double sigma = 30.0; // standard deviation for the Gaussian envelope
    double theta = 0.0; // orientation of the normal to the parallel stripes
    double lambda = 10.0; // wavelength of the cosine factor
    double gamma = 0.5; // spatial aspect ratio

    // Apply the Gabor filter to the input image
    Mat gbr = gaborFilter(ksize, sigma, theta, lambda, gamma);
    Mat dest; // destination matrix for the filtered image
    filter2D(img, dest, -1, gbr); // 2D filter applied to the input image

    vector<float> color_fvec; // vector to store color features
    vector<float> texture_fvec; // vector to store texture features

    // Initialize vectors for the 2D histogram of the gradient magnitude image
    int Hsize = bins; // number of bins for the histogram
    int dim[2] = { Hsize, Hsize }; // dimensions of the histogram
    Mat hist2d1, hist2d2; // matrices to store the histograms
    hist2d1 = Mat::zeros(2, dim, CV_32S); // initialize the first histogram matrix
    hist2d2 = Mat::zeros(2, dim, CV_32S); // initialize the second histogram matrix
    int i, j, c, sumr1, sumr2, sumg1, sumg2; // loop variables
    float r1, g1, b1, r2, g2, b2; // color values

    // Calculate the gradient magnitude image histogram
    for (i = 0; i < dest.rows; i++)
    {
        for (j = 0; j < dest.cols; j++)
        {
            // Retrieve the RGB values of a pixel
            r1 = dest.at<Vec3b>(i, j)[2];
            g1 = dest.at<Vec3b>(i, j)[1];
            b1 = dest.at<Vec3b>(i, j)[0];

            // Calculate the histogram bin indices based on the intensity values
            sumr1 = Hsize * r1 / (r1 + b1 + g1 + 1);
            sumg1 = Hsize * g1 / (r1 + b1 + g1 + 1);

            // Increment the histogram bin count
            hist2d1.at<int>(sumr1, sumg1)++;
        }
    }

    // Convert the histogram matrix to a vector
    for (i = 0; i < hist2d1.rows; i++)
    {
        for (j = 0; j < hist2d1.cols; j++)
        {
            texture_fvec.push_back(hist2d1.at<int>(i, j));
        }
    }

    // Calculating color histogram of image
    for (i = 0; i < img.rows; i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            // Extracting the red, green, and blue values of the current pixel
            r2 = img.at<Vec3b>(i, j)[2];
            g2 = img.at<Vec3b>(i, j)[1];
            b2 = img.at<Vec3b>(i, j)[0];

            // Calculating the indices for the red and green values in the 2D histogram
            sumr2 = Hsize * r2 / (r2 + b2 + g2 + 1);
            sumg2 = Hsize * g2 / (r2 + b2 + g2 + 1);

            // Incrementing the value at the corresponding bin in the histogram
            hist2d2.at<int>(sumr2, sumg2)++;
        }
    }

    // Flattening the 2D histogram into a 1D vector
    for (i = 0; i < hist2d2.rows; i++)
    {
        for (j = 0; j < hist2d2.cols; j++)
        {
            // Adding the histogram bin value to the color feature vector
            color_fvec.push_back(hist2d2.at<int>(i, j));
        }
    }

    // Concatenating the color and texture feature vectors into a single feature vector
    texture_fvec.insert(texture_fvec.end(), color_fvec.begin(), color_fvec.end());

    // Adding the texture-color feature vector to the final feature vector
    for (auto& n : texture_fvec)
    {
        fvec.push_back(n);
    }

    return 0;

}

int texturecolor_histx(vector<float>& target_data, vector<string> filename, vector<vector<float>>& fvec, int N)
{
    // initialize variables to store the intersection value and the sum of the values in the target image and directory images
    float intersection;
    double target_sum = 0;
    double dir_sum = 0;

    // initialize variables for normalized histograms of target and directory images
    float normalized_target;
    float normalized_dir;

    // initialize a vector to store the minimum values
    vector<float> all_min;

    // initialize a struct to store the image filename and histogram intersection value as a pair
    struct ImageDetail
    {
        string img_name;
        float value;
    };

    vector<ImageDetail> img_value;
    ImageDetail pair;

    // compute the sum of the values in the target image data
    target_sum = accumulate(target_data.begin(), target_data.end(), 0);

    // loop through the set of directory images
    for (int i = 0; i < filename.size(); i++)
    {
        // compute the sum of the values in the current directory image
        dir_sum = accumulate(fvec[i].begin(), fvec[i].end(), 0);

        float min_sum = 0;
        // loop through each bin of the target image
        for (int j = 0; j < target_data.size(); j++)
        {
            // normalize the values of the target and directory images at the current bin
            normalized_target = target_data[j] / target_sum;
            normalized_dir = fvec[i][j] / dir_sum;

            // compute the sum of all normalized minimum values
            min_sum += min(normalized_target, normalized_dir);
        }

        // compute the histogram intersection as the absolute difference from 1 of the sum of the minimum values
        intersection = abs(1 - min_sum);

        // store the histogram intersection value and image filename as a pair in the "img_value" vector
        pair.img_name = filename[i];
        pair.value = intersection;
        img_value.push_back(pair);
    }

    // sort the "img_value" vector in ascending order of histogram intersection values
    sort(img_value.begin(), img_value.end(), [](const ImageDetail& a, const ImageDetail& b)
        {
            return a.value < b.value;
        });

    // resize the "img_value" vector to the desired number of matches (N)
    img_value.resize(N);

    // print the top N-1 matches
    std::cout << "The top " << N - 1 << " matches are:" << std::endl;

    // loop through the "img_value" vector and print the image filename and histogram intersection value for each match
    int i = 0;
    for (auto& n : img_value) {
        if (i == 0)
        {
            i++;
            continue;
        }
        std::cout << n.img_name << ": " << std::fixed << n.value << std::endl;
    }
    return 0;
}


int main()
{
    // path to the target image file
    std::string targetpath = "/Users/illusionsoftruth/Downloads/olympus/pic.0210.jpg";

    // read the target image
    cv::Mat targetimg = cv::imread(targetpath);

    // vector to store the texture and color features of the target image
    std::vector<float> targetvec;

    // directory path for the images
    char dirname[] = "/Users/illusionsoftruth/Downloads/olympus";

    // buffer to store the complete file path of each image
    char buffer[256];

    // pointer to the directory
    DIR* dirp;

    // structure to store information about each file in the directory
    struct dirent* dp;

    // vectors to store the filenames and feature vectors of images in the directory
    std::vector<std::string> dir_filenames;
    std::vector<std::vector<float>> dir_fvec;

    // print the directory path
    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);

    // check if the directory was successfully opened
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // extract texture and color features from the target image
    texturecolor(targetimg, targetvec, 16);

    // loop over all the files in the directory
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image (based on the file extension)
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // build the complete file path
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // read the image
            cv::Mat img = cv::imread(buffer, cv::IMREAD_COLOR);

            // check if the image was successfully read
            if (img.data == NULL) {
                printf("Unable to read file %s, skipping\n", buffer);
                continue;
            }

            std::vector<float> fvec; // vector for image data
            texturecolor(img, fvec, 16);
            dir_filenames.push_back(dp->d_name);
            dir_fvec.push_back(fvec);


        }
    }
    texturecolor_histx(targetvec, dir_filenames, dir_fvec, 4);
    return 0;
}
