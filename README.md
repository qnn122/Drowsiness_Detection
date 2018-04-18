# Drowsiness Detection System
A simple software detecting user's drowsiness level by tracking eye activities. 

There are 4 main stages:
1. Face Detection - AdaBoost
2. Eyes Localization - Random Forest
3. Eye state monitoring - Various image processing techniques (global thresholding, image dilation
4. Drowsines Detection - Automatic threshold selection

The program was primarily implemented in MATLAB, some parts in C/C++ (mex file) to boost up the performance.

To starts the program, simply run DrowsinessDetectionGUI.m or simply type on command line:
    >> DrowsinessDetectionGUI
    
Click [here](https://youtu.be/YsL4wMvDNgI) to watch the demo video.
