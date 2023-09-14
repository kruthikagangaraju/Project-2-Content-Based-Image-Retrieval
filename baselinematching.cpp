/*Kruthika Gangaraju and Sriram Kodeeswaran */


#include <iostream> // For input/output operations
#include <vector>   // For using vector data structure
#include <algorithm> // For sorting the elements in vector
#include <dirent.h> // For reading the contents of a directory
#include <opencv2/core.hpp>  // OpenCV library for image processing
#include <opencv2/imgproc.hpp> // OpenCV library for image processing
#include <opencv2/imgcodecs.hpp> // OpenCV library for reading and writing images
#include <opencv2/highgui.hpp> // OpenCV library for GUI operations
#include <fstream>  // For reading and writing to/from files
#include <string>   // For using strings

using namespace cv;  // Using OpenCV namespace
using namespace std; // Using Standard namespace

// This function calculates the difference between two images by comparing the RGB values of corresponding pixels in a 9x9 region centered at the center of each image. The difference between the RGB values is squared and accumulated, and the cumulative difference is returned as the output of the function.
double calculateImageDistance(Mat targetImage, Mat databaseImage) {
    double distance = 0;
    int w = targetImage.rows / 2;
    int h = targetImage.cols / 2;
    
    // Loop over a 9x9 region in the center of the image
    for (int i = w-4; i <= w+4; i++)
    {
        for (int j = h-4; j <= h+4; j++)
        {
            for (int channel = 0; channel < 3; channel++) {
                // Accumulate the difference between corresponding RGB values
                distance += pow(targetImage.at<cv::Vec3b>(i, j)[channel] - databaseImage.at<cv::Vec3b>(i, j)[channel], 2);
            }
        }
    }
    return distance;
}

int main(int argc, char** argv) {
    // Target image file path
    string targetImagePath = "/Users/illusionsoftruth/Downloads/olympus/pic.1016.jpg";
    // Read the target image
    Mat target = imread(targetImagePath, IMREAD_COLOR);
    // Vector to store image distances and corresponding filenames
    vector<pair<double, string>> imageDistances;
    
    // Path to the directory containing the database images
    string databaseFolderPath = "/Users/illusionsoftruth/Downloads/olympus";
    // Vector to store the filenames of the database images
    vector<string> imageFilenames;
    
    // Open the directory containing the database images
    DIR* dir;
    struct dirent* ent;
    dir = opendir(databaseFolderPath.c_str());
    // If the directory could not be opened, print an error message and exit the program
    if (dir == NULL) {
        cout << "Error: Could not open directory " << databaseFolderPath << endl;
        return -1;
    }
    // Read the contents of the directory and store the filenames of the images in the vector
    while ((ent = readdir(dir)) != NULL) {
        string filename = ent->d_name;
        if (filename != "." && filename != ".." && filename != "pic.1016.jpg" && (filename.find(".jpg") != std::string::npos)) {
            imageFilenames.push_back(filename);
        }
    }
    closedir(dir);
    
    // Iterate through all the filenames in the imageFilenames vector
    for (string filename : imageFilenames) {
        // Construct the file path for each image file
        string filepath = databaseFolderPath + "/" + filename;
        // Read the database image using OpenCV's imread function
        Mat databaseImage = imread(filepath, IMREAD_COLOR);
        
        // Calculate the difference between the target image and the current database image
        double distance = calculateImageDistance(target, databaseImage);
        // Add the calculated difference and the filename to the imageDistances vector
        imageDistances.push_back(make_pair(distance, filename));
    }
    
    // Sort the imageDistances vector in ascending order based on the difference value
    sort(imageDistances.begin(), imageDistances.end());
    // Set N to 3 to find the top 3 closest images
    int N = 3;
    // Print the result
    cout << "The top " << N << " closest images are: " << endl;
    // Iterate through the top N closest images and print the filenames
    for (int i = 0; i < N; i++) {
        cout << imageDistances[i].second << endl;
        
    }
    // Return 0 to indicate the program has executed successfully
    return 0;
}
