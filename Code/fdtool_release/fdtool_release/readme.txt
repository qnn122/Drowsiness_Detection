
Objects/Faces detection toolbox v 0.26
--------------------------------------

This toolbox provides some tools for objects/faces detection using Local Binary Patterns (and some variants) and Haar features.
Object/face detection is performed by evaluating trained models over multi-scan windows with boosting models
(such Adaboosting, FastAdaboosting and Gentleboosting) or with linear SVM models.
The main objective of FDT is to bring simple but efficient tools mainly written in C codes with a matlab interface and easy to modify.  


BEFORE INSTALLATION, BE SURE TO HAVE A C COMPILER ON YOUR SYSTEM!!!!!

For windows system, recommanded compilers are MSVC/MSVC express (free)/Intel compiler
For Linux system, recommanded compilers are GCC(free)/Intel compiler

PLEASE BE SURE THAT YOU SETUP YOUR COMPILER BEFORE FDT INSTALLATION.

For checking, please type in matlab command : mex -setup 
and choose your favorite compiler

For Windows system, default LCC compiler included in matlab can't compile all files, you should have some errors during installation. 
Please use precompiled mex-files included in zip files (mexw32.zip/mexw64.zip respectively).

For Windows system, you may need also to add OMP_NUM_THREADS equal to the number of core in your system variables (if OpenMP failed)

For Linux system, you may need to install, the std++ package. Use the fowllowing command (Thanks to R. Mattheij) : 

$sudo apt-get install g++-multilib 

Installation
------------

This toolbox has been tested on Windows system and should also work for Linux plateform without any problem (Thanks to R. Mattheij for 
Linux testing, I don't have personaly a Linux box close to me).

------> Please run "setup_fdt" to install, compile each mex-files and add fdtool directory in the matlab path. 

Type "help mexme_fdt" for more compilation options.

Please open *.m or *.c files to read full description/instruction of each function and main references.

Run First
---------

a) Play with "demo_detector_haar.m" or "demo_detector_hmblbp.m" for real-time face tracking. 
   For windows system, you can use the included VCAPG2 webcam grabber. Otherwise and for Linux systems, you must have the IMAQ Toolbox (getsnapshot function).

b) View examples included in "train_cascade" for training Haar/MBLBP features with boosting algorithms and cascade (type "help train_cascade")

c) View examples included in "train_cascade_Xpos" for training Haar/MBLBP features with boosting algorithms and cascade (type "help train_cascade_Xpos")
   Positives examples are staked in a 3D tensors.

d) View examples included in "train_model" for training Haar/MBLBP/HMBLBP/HCSMBLBP features with boosting/SVM algorithms.
  (type "help train_model")


VCAPG2 webcam grabber for windows
---------------------------------

In order to compile vcapg2.cpp (webcam grabber for windows), please 
i) download the last Windows DDK at http://www.microsoft.com/downloads/en/details.aspx?displaylang=en&FamilyID=36a2630f-5d56-43b5-b996-7633f2ec14ff
ii) copy qedit.h into c:\Program Files\Microsoft SDKs\Windows\vx.x\Include folder where x.x designs the DDK version (currently last is 7.1)
iii) compile with mex vcapg2.cpp -I"c:\Program Files\Microsoft SDKs\Windows\vx.x\Include"

Thanks to Pr. Fehn for the x64 adaptation of vcapg2


Training set
------------

i) Positives examples for boosting approaches

Viola-Jones [4] and Jensen positives examples are included in this package in mat format, viola_24x24.mat and jensen_24x24.mat respectively where 
the size of each image is (24 x 24). 

ii) Positives examples for Histogram of feature + SVM

Please download a face database, for example the MIT-CBCL at:

http://cbcl.mit.edu/software-datasets/heisele/download/MIT-CBCL-facerec-database.zip

Extract "training-synthetic" pgm files in "positives" dir.

Or use the lfw cropped database available at http://itee.uq.edu.au/~conrad/lfwcrop/lfwcrop_grey.zip

P.S: CMU+MIT frontal faces dataset is available at: http://www.ee.oulu.fi/~jiechen/download/MIT-CMU-frontal-face-set-4-Timo.zip
     (since the official link seems broken)


iii) Negatives examples

A relative small "negatives" archive is also included and must be unpacked into "negatives" dir.
2 possibilities to retrieve more negatives pics in jpeg format:

a)use "build_negatives.m" function or
b)download the following zip file: http://c2inet.sce.ntu.edu.sg/Jianxin/RareEvent/nonface.zip and extract in "negatives" subfolder
 (be aware that there are still some positives faces in this zip !!!)



Demos
-----

10 demos are included. 

i)    "demo_mblbp"
ii)   "demo_chlbp"
iii)  "demo_haar"
vi)   "demo_detector_haar"
v)    "demo_detector_hmblbp"
vi)   "demo_fine_threshold.m"
vii)  "demo_haar_mblbp_training.m"
viii) "demo_mblbp.m"
ix)   "demo_mblbp_variant_training.m"
x)    "demo_type_cascade_scaling_vs_interp.m"
xi)   "demo_detector_hmblgp"



Organization
------------


     A) HAAR Features
            
            detector_haar                                        Real-Time face detector based on Haar's features trained with boosting methods
            eval_haar                                            Compute output of a trained strong classifier for new instances matrix X
            eval_haar_subwindow                                  Compute output of a trained strong classifier for new instances matrix X at different scale
            fast_haar_ada_weaklearner                            Train ffast a Decision stump weaklearner with adaboost on Haar features. 
                                                                 Assume normal pdf for positives ans negatives examples
            fast_haar_adaboost_binary_model_cascade              Train a strong classifier with Fasthaaradaboosting on Haar features
            gui_features_dictionary                              GUI for creating patterns dictionary
            haar                                                 Compute the Haar features for a given set of featured defined by haar_featlist
            haar_ada_weaklearner                                 Decision stump weaklearner for adaboosting on Haar features computed online
            haar_ada_weaklearner_memory                          Decision stump weaklearner for adaboosting on Haar features computed offline
            haar_adaboost_binary_train_cascade                   Train a strong classifier with Adaboosting on Haar features computed online
            haar_adaboost_binary_train_cascade_memory            Train a strong classifier with Adaboosting on Haar features computed offline
            haar_adaboost_binary_predict_cascade                 Predict label with trained model with Adaboosting on Haar features computed online
            haar_adaboost_binary_predict_cascade_memory          Predict label with trained model with Adaboosting on Haar features computed offline
            haar_featlist                                        Compute Haar features parameters
            haar_gentle_weaklearner                              Decision stump weaklearner for gentleboosting on Haar features computed online            
            haar_gentle_weaklearner_memory                       Decision stump weaklearner for gentleboosting on Haar features computed offline
            haar_gentleboost_binary_train_cascade                Train a strong classifier with Gentleboosting on Haar features computed online
            haar_gentleboost_binary_train_cascade_memory         Train a strong classifier with Gentleboosting on Haar features computed offline
            haar_gentleboost_binary_predict_cascade              Predict label with trained model with Gentleboosting on Haar features computed online
            haar_gentleboost_binary_predict_cascade_memory       Predict label with trained model with Gentleboosting on Haar features computed offline
            Haar_matG                                            Sparse Haar Features matrix for fasthaaradaboosting
            haar_scale                                           Haar features scaled to Faces database size

     B) CHLBP features

            chlbp                                                Compute the chlbp features
            chlbp_adaboost_binary_train_cascade                  Train a strong classifier with Adaboosting on chlbp features
            chlbp_adaboost_binary_predict_cascade                Predict label with trained model with Adaboosting on chlbp features
            chlbp_gentleboost_binary_train_cascade               Train a strong classifier with Gentleboosting on chlbp features
            chlbp_gentleboost_binary_predict_cascade             Predict label with trained model with Gentleboosting on chlbp features
            eval_chlbp                                           Compute output of a trained strong classifier for new instances matrix X

     C) MBLBP features
            
            detector_mblbp                                       Real-Time face detector based on mblbp's features trained with boosting methods
            eval_mblbp                                           Compute output of a trained strong classifier for a set of images of size (ny x nx)
            eval_mblbp_subwindows                                Compute output of a trained strong classifier for a new image of size (Ny x Nx)
            mblbp                                                Compute the mblbp features
            mblbp_ada_weaklearner                                Decision stump weaklearner for adaboosting on mblbp features
            mblbp_adaboost_binary_train_cascade                  Train a strong classifier with Adaboosting on mblbp features
            mblbp_adaboost_binary_predict_cascade                Predict label with trained model with Adaboosting on mblbp features
            mblbp_featlist                                       Compute mblbp features parameters
            mblbp_gentle_weaklearner                             Decision stump weaklearner for gentlboosting on mblbp features
            mblbp_gentleboost_binary_train_cascade               Train a strong classifier with Gentleboosting on mblbp features
            mblbp_gentleboost_binary_predict_cascade             Predict label with trained model with Gentleboosting on mblbp features

     D) HMBLBP_spyr & HMBLGP_spyr features

            detector_mlhmslbp_spyr                               Real-Time face detector based on histogram of LBP features trained with L-SVM method
            eval_hmblbp_spyr_subwindow                           Compute output of a trained classifier for new images
            detector_mlhmslgp_spyr                               Real-Time face detector based on histogram of LGP features trained with L-SVM method
            eval_hmblgp_spyr_subwindow                           Compute output of a trained classifier for new images


     E) MISCELLANEOUS 
           
            area                                                 Compute area of rectangular ROI with Integral Image method
            auroc                                                Compute the Area Under the ROC
            basicroc                                             Compute ROC given true label and Outputs of Strong classifiers 
            build_negatives                                      Download from internet set of images used to construct negatives subwindows
            display_database                                     Display all faces/non faces database
            eval_model_dataset                                   Evaluate trained model on a set of extracted Positives and Negatives pictures from positves and negatives folder respectively
            fast_rotate                                          Rotate UIUT8 grayscale image
            generate_data_cascade                                Generate positives features & negatives features for training a cascade with boosting methods
            generate_data_cascade_Xpos                           Generate positives features from 3D-stacked images & negatives features for training a cascade with boosting methods
            generate_face_features                               Generate positives & negatives features for training Histogram Integral features via large-scale SVM
            generate_nd_features                                 Generate positives non-detection examples passing through current trained large-scale SVM model
            generate_fa_features                                 Generate negatives false alarms features passing through current trained large-scale SVM model
            getmapping                                           Mapping feature's values for CHLBP and MBLBP approaches (from Marko Heikkilä and Timo Ahonen LBP toolbox)
            ieJPGSearch                                          Included michaelB function to retrive URL links of images from Google
            image_integral_standard                              Standardize and compute Images Integral
            image_standard                                       Standardize images
            imresize                                             Resize UINT8 images by bilinear interpolation
            int8tosparse                                         Convert a int matrix in a sparse matrix (thx J. Tursa)
            inv_integral_image                                   Retrieve images from images integral
            jensen_24x24                                         Ole Jensen faces/non faces database
            mexme_fdt                                            Script for compiling mex files
            nbfeat_haar                                          Compute number of features given image database size and patterns
            plot_rectangle                                       Display rectangles associated with detected faces
            perf_dr_pfa                                          Compute detection rate versus number of false alarm given set of images with ground truth face locations
            rgb2gray                                             Convert RGB image in gray format
            train_cascade                                        Train cascade model with coventional/multi-exit asymetric boosting approach, positives & negatives are in their respective folder  
            train_cascade_Xpos                                   Train cascade model with coventional/multi-exit asymetric boosting approach with positive examples stacked in a 3D tensor 
            train_dense                                          Liblinear fast Linear SVM solver with dense input format support
            train_model                                          Train model for (Haar,MBLBP,HMBLBP/HCSMBLBP) features via (Adaboosting/Gentleboosting/Linear SVM (liblinear)) with eventually a positives & negatives boosting
            train_stage_cascade                                  Train a stage of the cascade for boosting method
            vcapg2                                               Capture webcam frames (see  Kazuyuki Kobayashi file from http://www.mathworks.com/matlabcentral/fileexchange/2939 )
            viola_24x24                                          Viola-Jones faces/non faces database
            setup_fdt                                            Install and setup the face detection toolbox


 Author  Sébastien PARIS : sebastien.paris@lsis.org   for contact and bugs reporting 
 ------  Initial release date : 02/20/2009



 Main References         [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------------             The annals of statistics, 1999

                         [2] Zhang, L. and Chu, R.F. and Xiang, S.M. and Liao, S.C. and Li, S.Z, "Face Detection Based on Multi-Block LBP Representation"
                             ICB07

                         [3] C. Huang, H. Ai, Y. Li and S. Lao, "Learning sparse features in granular space for multi-view face detection", FG2006
 
                         [4] P.A Viola and M. Jones, "Robust real-time face detection", International Journal on Computer Vision, 2004

                         [5] M-T. Pham and all, "Detection with multi-exit asymetric boosting", CVPR'08 

                         [6] Eanes Torres Pereira, Herman Martins Gomes, João Marques de Carvalho 
                             "Integral Local Binary Patterns: A Novel Approach Suitable for Texture-Based Object Detection Tasks"
                             2010 23rd SIBGRAPI Conference on Graphics, Patterns and Images

                         [7] Martijn Reuvers, "Face Detection on the INCA+"
                             http://www.science.uva.nl/research/ias/alumni/m.sc.theses/theses/MartijnReuvers.pdf

                         [8] Sebastien Paris, Herve Glotin and Zhong-Qiu Zhao,
                             "Real-time face detection using Integral Histogram of Multi-Scale Local Binary Patterns"
                             ICIC 2011 



Greetings to    i )  Ole Jensen for providing me his faces database and the merging detections algorithm for detector_haar and detector_mblbp,  
------------    ii)  Pham Minh Tri for his responses concerning Fastadaboosting and multi-exit asymetric boosting,
                iii) Pr Fehn for his modified version of vcapg2.
                iv)  Zhu Jianqing for his conventional cascade and for provinding to me the URL for negatives picts


Changelogs 
----------

v0.26 08/12/12  Minor update
                - Fix all functions with spyr variable. Now spyr matrix are (nscale x 5) instead of (nscale x 4)
                - Fix train_cascade

v0.25 12/31/11  Minor update
                - Correct a bug in eval_hmblbp_spyr_subwindow.c and detector_mlhmslbp_spyr.c with the cs_opt option 
                - Correct bugs in mexme_fdt for Linux system
                - Correct comments in haar.c to be compiled for Linux system
                - Fix description of input/ouput in Haar_featlist and dependencies (thanks to Lucas Chai)
                - Fix train_model (fxtemp problem).
                - Miscalleneous changes

v0.24 11/16/11  Minor update:
                - Correct bugs in eval_hmblbp_spyr_subwindow.c
                - Minor comestic changes
                - Update readme.txt

v0.23 11/09/11  Minor update:
                - Update spyr option for HMSLBP approach. Now weights of each subwindows can be tuned by users.
                - Add online help on detector functions
             
v0.22 05/16/11  Minor update:
                - Add new normalization option for detector_mlhmslbp_spyr.c and eval_hmblbp_spyr_subwindow.c. Now feature vectors can be normalized for each
                  subwindows and/or for each subwindows in a current pyramid level and/or for the full vector. When using homogeneous kernel
                  such chi2, Intersection histogram (with options.n > 0), feature vectors should be L1-normalized (options.norm = [1 , x , x ]). If options.n = 0,
                  Features vectors can be only locally normalize (options.nom = [0 , 0 , x]) or/and eventully also L2 fully normalized (options.nom = [2 , 0 , x])
                - Add seed option to generate the same posives/negative sets
                - Group pictures in "images/train" or "image/test" folder
                - Can work with several file extensions to build positive/negative sets.
                - train_cascade now uses two spectific positives & negatives folders (use it if u want to train something else than faces). 
                  train_cascade_Xpos uses 3D stacked positives matrix.
                - Rename some functions and remove unecessary files
                - Add eval_mblbp_subwindows.c
                - Fix small bugs and typos
                - Add perf_dr_fa.m (beta) to evaluate detector performances.
                - Update readme.txt

v0.21 04/11/10  Minor update:
                - Add missing negatives.zip
                - Add dense version of Liblinear
                - Fix chlbp.c


v0.2 03/11/10   Major Update:
                - Changed Input parsing, only (1+1) imputs, a unique options/model structure must be given right now as imput 
                - Inputs parsing more flexible and with more options
                - OpenMP support (Multicore) for faster training and detection
                - Better and faster cascade training algorithms (conventional and multi-exit cascade)
                - Better merging detections (removing internal subwindows inside a larger one)
                - Add haar_gentleboost_binary_train_cascade_memory.c, haar_gentleboost_binary_predict_cascade_memory.c, haar_gentle_weaklearner_memory.c
                  haar_adaboost_binary_train_cascade_memory.c, haar_adaboost_binary_predict_cascade_memory.c and haar_ada_weaklearner_memory.c 
                  for fast and exact Haar training but for system with enough RAM (at least 8gb of RAM is required, see options.algoboost = 3,4)
                - Fix a crash for FastHaarAdaboosting with 64 bits systems (add largeArrayDims option for sparse matrix)
                - Add fine_threshold option for FastHaarAdaboosting improving a little accuracy
                - Add tranpose option to speedup weaklearner training with pre-computed MBLBP and Haar features
                - Add probarotIpos, m_angle and sigma_angle for rotate positives examples with angle~N(m_angle, sigma_angle²) in order to be more robust
                  to face orientation during training phase
                - Add a novel approach based on Fast computation of Histograms of features (Histogram Integral) (LBP here) and trained by Linear SVM (still in early stage of testing)
                - Correct, add some comments and fix some typos (there are still a lot I know).


v0.1-0.1f       - Fix bugs in fast_haar_ada_weaklearner and fast_haar_adaboost_binary_model_cascade when alpha = 0 (Thx to Zhu), minor cleanup.
                - Fix some small bugs, better Linux64 support, include vcapg2 modified for win64.
                - Correct minor crashes
                - Should correctly compile with LCC and prior release of Matlab (R13 and upper versions) (thx to Bruno Luong for his function)
                - Should compile on Win64 platerform (thx to Soeren Sproessig for report bug)
                - Typo corrections 

To Do
-----       
             - A GOOD AND DETAILED DOCUMENTATION WITH NUMEROUS EXAMPLES 
             - Mix Haar/MLBP features for boosting approaches
             - Change the weaklearner in the MBLBP + boosting approach (use table lookup instead or DT)
             - Add optimized weights for Haar features
             - Multi-thresholds weaklearner
             - Add MBLDP (for MultiBlock Local Derivative Patterns) features for boosting approaches
             - Add other Histograms features: HOG, HMBLDP,etc...
             - Add multi-class support
             - Etc ... 

