/* Texture and Color Histogram
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

using namespace std;
using namespace cv;

int sobel_x(Mat& src, Mat& dst)
{
    int kernel1[] = { -1, 0, 1 }; // Define the kernel for the Sobel operator in the x-direction. The values represent the horizontal gradient.
    int kernel2[] = { 1, 2, 1 }; // Define the kernel for the Sobel operator in the y-direction. The values represent the vertical gradient.
    
    Mat temp(src.size(), CV_16SC3); // Create a temporary matrix to store intermediate results.
    for (int i = 1; i < src.rows - 1; i++) // Loop through the rows of the source image. Start from the second row to the second-last row.
    {
        for (int j = 1; j < src.cols - 1; j++) // Loop through the columns of the source image. Start from the second column to the second-last column.
        {
            int sb1 = 0; // Initialize the blue channel gradient sum for the x-direction.
            int sg1 = 0; // Initialize the green channel gradient sum for the x-direction.
            int sr1 = 0; // Initialize the red channel gradient sum for the x-direction.
            for (int k = i - 1; k <= i + 1; k++) // Loop through the three rows centered at the current row.
            {
                Vec3b sbly = src.at<Vec3b>(k, j); // Get the blue, green, and red values of the current pixel.
                sb1 = sb1 + sbly[0] * kernel1[k - (i - 1)]; // Multiply the blue value by the corresponding kernel value and add it to the blue channel gradient sum.
                sg1 = sg1 + sbly[1] * kernel1[k - (i - 1)]; // Multiply the green value by the corresponding kernel value and add it to the green channel gradient sum.
                sr1 = sr1 + sbly[2] * kernel1[k - (i - 1)]; // Multiply the red value by the corresponding kernel value and add it to the red channel gradient sum.
            }
            temp.at<Vec3s>(i, j)[0] = abs(sb1 / 4); // Store the absolute value of the blue channel gradient sum for the x-direction divided by 4 in the temporary matrix.
            temp.at<Vec3s>(i, j)[1] = abs(sg1 / 4); // Store the absolute value of the green channel gradient sum for the x-direction divided by 4 in the temporary matrix.
            temp.at<Vec3s>(i, j)[2] = abs(sr1 / 4); // Store the absolute value of the red channel gradient sum for the x-direction divided by 4 in the temporary matrix.
        }
    }
    
    for (int i = 1; i < temp.rows - 1; i++)
    {
        for (int j = 1; j < temp.cols - 1; j++)
        {
            int sb2 = 0; //Variable to store sum of blue channel values
            int sg2 = 0; //Variable to store sum of green channel values
            int sr2 = 0; //Variable to store sum of red channel values
            
            for (int k = j - 1; k <= j + 1; k++)
            {
                Vec3s sbly = temp.at<Vec3s>(i, k);
                int filter_point = kernel2[k - (j - 1)];
                sb2 = sb2 + sbly[0] * kernel2[k - (j - 1)];
                sg2 = sg2 + sbly[1] * kernel2[k - (j - 1)];
                sr2 = sr2 + sbly[2] * kernel2[k - (j - 1)];
            }
            dst.at<Vec3s>(i, j)[0] = abs(sb2); //abs function is used to calculate absolute values
            dst.at<Vec3s>(i, j)[1] = abs(sg2);
            dst.at<Vec3s>(i, j)[2] = abs(sr2);
        }
    }
    return 0;
}

int sobel_y(Mat& src, Mat& dst)// Perform the Sobel operation in the Y direction
{
    int kernel1[] = { 1, 2, 1 }; // Horizontal kernel
    int kernel2[] = { -1, 0, 1 }; // Vertical kernel
    Mat temp(src.size(), CV_16SC3); // Create a temporary matrix to store intermediate results
    // Loop through the image rows (excluding border pixels)
    for (int i = 1; i < src.rows - 1; i++)
    {
        // Loop through the image columns (excluding border pixels)
        for (int j = 1; j < src.cols - 1; j++)
        {
            int sb1 = 0; // Initialize blue channel sum to 0
            int sg1 = 0; // Initialize green channel sum to 0
            int sr1 = 0; // Initialize red channel sum to 0
            
            // Convolve the source image with the horizontal kernel
            for (int k = i - 1; k <= i + 1; k++)
            {
                Vec3b sblx = src.at<Vec3b>(k, j); // Get the blue, green, and red channel values for the current pixel
                sb1 = sb1 + sblx[0] * kernel1[k - (i - 1)]; // Multiply the blue channel value with the corresponding kernel value and add to the sum
                sg1 = sg1 + sblx[1] * kernel1[k - (i - 1)]; // Multiply the green channel value with the corresponding kernel value and add to the sum
                sr1 = sr1 + sblx[2] * kernel1[k - (i - 1)]; // Multiply the red channel value with the corresponding kernel value and add to the sum
            }
            // Store the absolute values of the sums in the temporary matrix
            temp.at<Vec3s>(i, j)[0] = abs(sb1);
            temp.at<Vec3s>(i, j)[1] = abs(sg1);
            temp.at<Vec3s>(i, j)[2] = abs(sr1);
        }
    }
    
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 1; j < temp.cols - 1; j++) {
            int sb2 = 0; //sum of blue color intensity value of pixels in the neighborhood of the current pixel
            int sg2 = 0; //sum of green color intensity value of pixels in the neighborhood of the current pixel
            int sr2 = 0; //sum of red color intensity value of pixels in the neighborhood of the current pixel
            for (int k = j - 1; k <= j + 1; k++) {
                Vec3s sblx = temp.at<Vec3s>(i, k);
                //multiply the intensity value of the pixel with the corresponding weight from the vertical kernel matrix
                sb2 = sb2 + sblx[0] * kernel2[k - (j - 1)];
                sg2 += sg2 + sblx[1] * kernel2[k - (j - 1)];
                sr2 += sr2 + sblx[2] * kernel2[k - (j - 1)];
            }
            //divide the sum of intensity values by 4 to normalize the result
            dst.at<Vec3s>(i, j)[0] = abs(sb2 / 4);
            dst.at<Vec3s>(i, j)[1] = abs(sg2 / 4);
            dst.at<Vec3s>(i, j)[2] = abs(sr2 / 4);
        }
    }
    //return 0 to indicate success
    return 0;

}

int gradMagnitude(Mat& srcx, Mat& srcy, Mat& dst)
{
    for (int i = 0; i < srcx.rows; i++) // loop through all rows of srcx image
        {
            for (int j = 0; j < srcx.cols; j++) // loop through all columns of srcx image
                {
                    Vec3s x = srcx.at<Vec3s>(i, j); // get the intensity values for each channel (BGR) for pixel (i, j) in srcx image
                    Vec3s y = srcy.at<Vec3s>(i, j); // get the intensity values for each channel (BGR) for pixel (i, j) in srcy image



                    for (int c = 0; c < 3; c++)
                    {
                        dst.at<Vec3s>(i, j)[c] = sqrtf(pow(x[c], 2) + pow(y[c], 2));//using sobelx and sobely values to calculate
            }
        }
    }
    return 0;
}

int texturecolor(Mat& img, vector<float>& fvec, int bins)
{
    
    /* TEXTURE */
    // apply sobel filters, gradient magnitude filter, and greyscale filter
    cv::Mat sx(img.size(), CV_16SC3);
    sobel_x(img, sx);

    cv::Mat sy(img.size(), CV_16SC3);
    sobel_y(img, sy);

    cv::Mat mag_img(img.size(), CV_16SC3);
    gradMagnitude(sx, sy, mag_img);
    convertScaleAbs(mag_img, mag_img);

    vector<float> color_fvec;
    vector<float> texture_fvec;

    // initialize vectors for gray magnitude histogram
    int Hsize = bins;
    int dim[2] = { Hsize, Hsize };
    Mat hist2d1, hist2d2;
    hist2d1 = Mat::zeros(2, dim, CV_32S);
    hist2d2 = Mat::zeros(2, dim, CV_32S);
    int i, j, c, sumr1, sumr2, sumg1, sumg2;
    //int num_bins = 16;
    float r1, g1, b1, r2, g2, b2;

    for (i = 0; i < mag_img.rows; i++) // calculating grad mag image histograms
    {
        for (j = 0; j < mag_img.cols; j++)
        {
            r1 = mag_img.at<Vec3b>(i, j)[2];
            g1 = mag_img.at<Vec3b>(i, j)[1];
            b1 = mag_img.at<Vec3b>(i, j)[0];
            sumr1 = Hsize * r1 / (r1 + b1 + g1 + 1);
            sumg1 = Hsize * g1 / (r1 + b1 + g1 + 1);
            hist2d1.at<int>(sumr1, sumg1)++;
        }
    }
    for (i = 0; i < hist2d1.rows; i++)
    {
        for (j = 0; j < hist2d1.cols; j++)
        {
            texture_fvec.push_back(hist2d1.at<int>(i, j));
        }
    }
    for (i = 0; i < img.rows; i++) // calculating color histogram
    {
        for (j = 0; j < img.cols; j++)
        {
            r2 = img.at<Vec3b>(i, j)[2];
            g2 = img.at<Vec3b>(i, j)[1];
            b2 = img.at<Vec3b>(i, j)[0];
            sumr2 = Hsize * r2 / (r2 + b2 + g2 + 1);
            sumg2 = Hsize * g2 / (r2 + b2 + g2 + 1);
            hist2d2.at<int>(sumr2, sumg2)++;
        }
    }
    for (i = 0; i < hist2d2.rows; i++)
    {
        for (j = 0; j < hist2d2.cols; j++)
        {
           color_fvec.push_back(hist2d2.at<int>(i, j));
        }
    }
    texture_fvec.insert(texture_fvec.end(), color_fvec.begin(), color_fvec.end()); // concatenating texture and color vectors

    for (auto& n : texture_fvec)
    {
        fvec.push_back(n);
    }

    return 0;
}

int texturecolor_histx(vector<float>& target_data, vector<string> filename, vector<vector<float>>& fvec, int N)
{
    // initialize variables
    float intersection;
    double target_sum = 0;
    double dir_sum = 0;

    // initialize variables for normalized histogram
    float normalized_target;
    float normalized_dir;

    // initialize vector for min values
    vector<float> all_min;

    // initialize struct for image filename & value pairs
    struct ImageDetail
    {
        string img_name;
        float value;
    };

    vector<ImageDetail> img_value;
    ImageDetail pair;

    // compute target image data sum
    target_sum = accumulate(target_data.begin(), target_data.end(), 0);

    // find the min, normalize min image histograms, and populate values to vector
    for (int i = 0; i < filename.size(); i++)
    {
        dir_sum = accumulate(fvec[i].begin(), fvec[i].end(), 0);

        float min_sum = 0;
        for (int j = 0; j < target_data.size(); j++)
        {
            // normalize target and directory image values
            normalized_target = target_data[j] / target_sum;
            normalized_dir = fvec[i][j] / dir_sum;

            // compute the sum of all normalized minimum values
            min_sum += min(normalized_target, normalized_dir);
        }

        // compute the histogram intersection
        intersection = abs(1 - min_sum);

        // push intersection values & image filename as pairs to a vector
        pair.img_name = filename[i];
        pair.value = intersection;
        img_value.push_back(pair);
    }

    // sort ssd values from min to max
    sort(img_value.begin(), img_value.end(), [](const ImageDetail& a, const ImageDetail& b)
        {
            return a.value < b.value;
        });

    img_value.resize(N); // resizing image paired values for N size for easy extraction

   
    std::cout << "The top " << N - 1 << " matches are:" << std::endl;
//    for (auto& n : img_value)
//    {
//        std::cout << n.img_name << ": " << std::fixed << n.value << std::endl;
//    }
    int i=0;
    for (auto& n : img_value) {
        if(i==0)
        {i++;
            continue;}
        std::cout << n.img_name  << ": " << std::fixed << n.value << std::endl;
    }
    return 0;

    
}

int main()
{
    string targetpath = "/Users/illusionsoftruth/Downloads/olympus/pic.0535.jpg";
    Mat targetimg = imread(targetpath);
    vector<float> targetvec;
    char dirname[] = "/Users/illusionsoftruth/Downloads/olympus";
    char buffer[256];
   
    DIR* dirp;
    struct dirent* dp;
    
    vector<string> dir_filenames;
    vector<vector<float>> dir_fvec;

    // get the directory path
    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    texturecolor(targetimg, targetvec, 16);

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // read each image
            cv::Mat img;
            img = cv::imread(buffer, cv::IMREAD_COLOR);
            if (img.data == NULL) {
                printf("Unable to read file %s, skipping\n", buffer);
                continue;
            }

            std::vector<float> fvec; // vector for image data
            texturecolor(img, fvec, 16);
            
            if (dirname != "." && dirname != ".." && dirname != "pic.0535.jpg" )
                {
                dir_filenames.push_back(dp->d_name);
                dir_fvec.push_back(fvec);
            }


        }
    }
    texturecolor_histx(targetvec, dir_filenames, dir_fvec, 4);
    return 0;
}
