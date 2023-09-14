1. Operating system: macOS

2. IDE: Xcode

3. Each task is in a separate file, it needs to be run separately. The path for directory and the target image has to be updated according to the user.
For directory the path has to be put in char dirname[] and the target image path has to be put in string targetpath.

4. Each task has 2 functions : one for calculating the feature vector and other for calculating distance metric. These functions are then called in main function.

5. The texture and color histogram output images are those which closely resemble the target image in colors and textures.

6. For the 5th task, where we had to create a custom feature, we decided to match a template image with the database of images. 
The output had images with features that closely resembled the target image.
We choose an image with a shoe in it as the template and we found our results satisfactory as most of the images were either a shoe or white object in center.
 
7. For the extensions, we implemented Gabor filter to get the textures and proceeded in same manner as for texture and color histograms.

DISCLAIMER: The Texture and Color, Gabor Filter files take a long time to run.