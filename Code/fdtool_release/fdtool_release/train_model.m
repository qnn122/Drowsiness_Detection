function [options , model] = train_model(options)

%
% Train features (Haar,MBLBP,HMBLBP/HCSMBLBP,HMBLGP/HCSMBLGP) model via
% (Adaboosting/Gentleboosting/Linear SVM (liblinear))
% with eventually a positives & negatives boosting of hard examples
%
%
% Usage
% ------
%
% [options , model] = train_model(options)
%
%
% Input
% ------
%
%  options          Options struture
%
%                                            Exactracting positives & negatives
%
%                   positives_path   Path from positives images are loaded for generating positives examples
%                   negatives_path   Path from negative images are loaded for generating negative examples
%                   posext           Positive'extension files
%                   negext           Negative'extension files
%                   negmax_size      Maximum side of each negative image (default = 1000)
%                   seed             Seed value for random generator in order to generate the same positive/negative sets
%                   resetseed        Reset generator with given seed (default resetseed = 1)
%                   preview          Preview current example (default preview = 0)
%                   standardize      Standardize images (default standardize = 1)
%                   Npostrain        Number of desired positives examples for training set (Npostrain+Npostest <= Npostotal)
%                   Nnegtrain        Number of desired negatives examples for training set. Extracted by bilinear interpolation
%                   Npostest         Number of desired positives examples for testing set (Npostrain+Npostest <= Npostotal)
%                   Nnegtest         Number of desired negatives examples for testing set. Extracted by bilinear interpolation
%                   Nposboost        Number of non-detection (for positives examples passing through detector) to add in positives set (default Nposboost = 50)
%                   Nnegboost        Number of false alarms (for negatives examples passing through detector) to add in negatives set (default Nnegboost = 10000)
%                   boost_ite        Number of iteration for boosting examples (default boost_ite = 10)
%                   probaflipIpos    Probability to flip Positive examples (default probaflipIpos = 0.5)
%                   probarotIpos     Probability to rotate Positives examples with an angle~N(m_angle,sigma_angle) (default probarotIpos = 0.01)
%                   m_angle          Mean rotation angle value in degree (default mangle = 0)
%                   sigma_angle      variance of the rotation angle value (default sigma_angle = 5^2)
%                   probaswitchIneg  Probability to swith from another picture in the negatives database (default probaswitchIneg = 0.005)
%                   posscalemin      Minimum scaling factor to apply on positives patch subwindows (default scalemin = 0.25)
%                   posscalemax      Maximum scaling factor to apply on positives patch subwindows (default scalemax = 2)
%                   negscalemin      Minimum scaling factor to apply on negatives patch subwindows (default scalemin = 1)
%                   negscalemax      Maximum scaling factor to apply on negatives patch subwindows (default scalemax = 5)
%
%                                             Features
%
%                   typefeat         Type of features (featype: 0 <=> Haar, 1 <=> MBLBP, 2 <=> Histogram of MBLBP, 3 <=> Histogram of MBLGP)
%
%                   dimsItraining    Positive size for training
%                   rect_param       Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
%                   F                Features's list (6 x nF) in UINT32 where nF designs the total number of Haar features
%                   usesingle        Output in single format if usesingle = 1 (default usesingle = 0)
%                   transpose        Transpose Output if tranpose = 1 (in order to speed up Boosting algorithm, default tranpose = 0)
%
%                   n                Order approximation for homogeneous additive Kernel
%                   L                Sampling step (default L = 0.5);
%                   kerneltype       0 for intersection kernel, 1 for Jensen-shannon kernel, 2 for Chi2 kernel (default kerneltype = 0)
%                   numsubdiv        Number of subdivisions (default numsubdiv = 8);
%                   minexponent      Minimum exponent value (default  minexponent = -20)
%                   maxexponent      Maximum exponent value (default minexponent = 8)
%                   spyr             Spatial Pyramid (nspyr x 5) (default [1 , 1 , 1 , 1 , 1] with nspyr = 1)
%                                    where spyr(i,1) is the ratio of ny in y axis of the blocks at level i (by(i) = spyr(i,1)*ny)
%                                    where spyr(i,2) is the ratio of nx in x axis of the blocks at level i (bx(i) = spyr(i,3)*nx)
%                                    where spyr(i,3) is the ratio of ny in y axis of the shifting at level i (deltay(i) = spyr(i,2)*ny)
%                                    where spyr(i,4) is the ratio of nx in x axis of the shifting at level i (deltax(i) = spyr(i,4)*nx)
%                                    where spyr(i,5) is the weight's histogram associated to current level pyramid (w(i) = spyr(i,1)*spyr(i,2))
%                                    total number of subwindows nH = sum(floor(((1 - spyr(:,1))./(spyr(:,3)) + 1)).*floor((1 - spyr(:,2))./(spyr(:,4)) + 1))
%                   scale            Multi-Scale vector (1 x nscale) (default scale = 1) where scale(i) = s is the size's factor to apply to each 9 blocks
%                                    in the LBP computation, i = 1,...,nscale
%                   cs_opt           Center-Symetric LBP/LGP : 1 for computing CS-MBLBP/ CS-MBLGP features, 0 : for MBLBP/MBLGP (default cs_opt = 0)
%                   improvedLBP      0 for default 8 bits encoding LBP, 1 for 9 bits encoding (8 + central area)
%                   improvedLGP      0 for default 8 bits encoding LGP, 1 for 9 bits encoding (8 + central area)
%                   rmextremebins    Force to zero bin = {0 , {255,58,9}} if  rmextremebins = 1 (default rmextremebins = 1)
%                   norm             Normalization vector (1 x 3) : [for all subwindows, for each subwindows of a pyramid level, for each subwindows]
%                                    norm = 0 <=> no normalization, norm = 1 <=> v=v/(sum(v)+epsi), norm = 2 <=> v=v/sqrt(sum(v²)+epsi²),
%                                    norm = 3 <=> v=sqrt(v/(sum(v)+epsi)) , norm = 4 <=> L2-clamped (default norm = [0 , 0 , 4])
%                   clamp            Clamping value (default clamp = 0.2)
%                   maptable         Mapping table for LBP codes. maptable = 0 <=> normal LBP = {0,...,255} (default),
%                                    maptable = 1 <=> uniform LBP = {0,...,58}, maptable = 2 <=> rotation-invariant LBP = {0,...,35},
%                                    maptable = 3 <=> uniform rotation-invariant LBP = {0,...,9}
%
%                                                  Detector
%
%                   postprocessing   Type of postprocessing in order to reduce false alarms (default postprocessing = 1):
%                                    0: no postprocessing, i.e. raw detections, 1: merging if rectangles overlapp more than 25%
%                                    2 : Better Merging detections algorithm with parameters defined by mergingbox
%                   dimsIscan        Initial Size of the scanning windows, i.e. (ny x nx ) (default dimsIscan = [24 , 24])
%                   scalingbox       [scale_ini , scale_inc , step_ini] where :
%                                    scale_ini is starting scale factor for each subwindows to apply from the size of trained images (default scale_ini = 2)
%                                    scale_inc is Increment of the scale factor (default scale_inc = 1.4)
%                                    step_ini  is the overlapping subwindows factor such that delta = Round(step_ini*scale_ini*scale_inc^(s)) where s in the number of scaling steps (default step_ini = 2)
%                   mergingbox       [overlap_same , overlap_diff , step_ini]
%                                    overlap_same is the overlapping factor for merging detections of the same size (first step) (default overlap_same = 1/2)
%                                    overlap_diff is the overlapping factor for merging detections of the different size (second step) (default overlap_diff = 1/2)
%                                    dist_ini is the size fraction of the current windows allowed to merge included subwindows (default dist_ini = 1/3)
%                   max_detections   Maximum number of raw subwindows detections (default max_detections = 500)
%                   num_threads      Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)
%
%
%                                                  Classifier
%
%                   typeclassifier   Type of classifier (0<=>adaboosting, 1 <=> Gentleboosting, 2<=> LSVM (Liblinear)
%                   addbias          Add bias or not for model prediction (1/0) (default addbias = 1)
%                   s                Type of SVM classifier according of liblinear documentation (default s = 2)
%                   c                Cost parameter for Libnear (default c = 2)
%
%                   T                Number of weaklearners (default T = 50)
%
%
% Output
% ------
%
% options          Extended input options struture
%
%                  w                 Trained weights model (1 x (1 x ((1+improvedLBP)*Nbins*nH*nscale+addbias))
%			                         where Nbins = ([256,59,36,10]*(improvedLBP+1)) if cs_opt = 0, Nbins = ([16,15,10,10]*(improvedLBP+1)) if cs_opt = 1
%                  auc_est           Area Under the Curve of the test data
%                  tp                True positive rate for test data
%                  fp                False positive rate for test data
%                  fxtest            Model prediction values for test data (1 x Ntest)
%                  ytest_est         Predicted labels for test data (1 x Ntest)
%                  tpp               True positive rate for different threshold
%                  fpp               False positive rate for different threshold
%
% model            model structure for detector_mlhmslbp_spyr or eval_hmblbp_spyr_subwindow
%
% You must have positive examples in "positves" folder.
% For example, you can download one database at
% http://cbcl.mit.edu/software-datasets/heisele/download/MIT-CBCL-facerec-database.zip
% and extract "training-synthetic" pgm files in "positives" dir.
% or use the lfw cropped database available at http://itee.uq.edu.au/~conrad/lfwcrop/lfwcrop_grey.zip
%
% Negatives examples can be retrieved via the build_negatives.m function
%
%
%
%% Example 1 HMBLBP + LSVM %%
%
% close all
% %options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
% %options.posext             = {'png'};
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.negmax_size        = 1000;
% options.Npostrain          = 10000;
% options.Nnegtrain          = 10000;
% options.Npostest           = 10000;
% options.Nnegtest           = 10000;
% options.Nposboost          = 50;
% options.Nnegboost          = 1000;
% options.boost_ite          = 5;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.standardize        = 1;
% options.preview            = 0;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.05;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.probaswitchIneg    = 0.9;
% options.posscalemin        = 0.4;
% options.posscalemax        = 1.5;
% options.negscalemin        = 0.5;
% options.negscalemax        = 1.1;
% options.addbias            = 1;
% options.max_detections     = 5000;
%
% options.typefeat           = 2;
% options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
% options.scale              = [1];
% options.maptable           = 0;
% options.cs_opt             = 1;
% options.improvedLBP        = 0;
% options.rmextremebins      = 0;
% options.color              = 0;
% options.norm               = [0 , 0 , 2];
% options.clamp              = 0.2;
% options.n                  = 0;
% options.L                  = 1.2;
% options.kerneltype         = 0;
% options.numsubdiv          = 8;
% options.minexponent        = -20;
% options.maxexponent        = 8;
%
% options.dimsItraining      = [24 , 24];
% options.rect_param         = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
% options.usesingle          = 1;
% options.transpose          = 0;
%
% options.typeclassifier     = 2;
% options.s                  = 2;
% options.B                  = 1;
% options.c                  = 2;
% options.T                  = 50;
%
% options.num_threads        = -1;
% options.dimsIscan          = [24 , 24];
% options.scalingbox         = [2 , 1.4 , 1.8];
% options.mergingbox         = [1/2 , 1/2 , 1/3];
%
% [options , model]          = train_model(options);
%
% figure(1)
% plot(options.fpp , options.tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , options.perftest , options.auc_est))
% axis([-0.02 , 0.3 , 0.75 , 1.02])
%
% figure(2)
% plot(model.w)
%
% figure(3)
% plot(options.fpp , options.tpp  , 'linewidth' , 2)
% axis([-0.02 , 1.02 , -0.02 , 1.02])
% %legend('Cascade' , 'MultiExit', 'Full', 'Location' , 'SouthEast')
% grid on
% title(sprintf('ROC for HMBLBP + Linear SVM'))
%
% figure(4)
% semilogy(1:options.boost_ite , options.pd_per_stage , 1:options.boost_ite , options.pfa_per_stage , 'linewidth' , 2)
% legend('P_d' , 'P_{fa}' , 'location' , 'southwest')
%
% aa                       = vcapg2(0,3);
% min_detect               = 2;%120;
% figure(5);set(1,'doublebuffer','on');
% while(1)
%     t1   = tic;
%     aa   = vcapg2(0,0);
%     pos  = detector_mlhmslbp_spyr(rgb2gray(aa) , model);
%     image(aa);
%     hold on
%     h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
%     hold off
%     t2   = toc(t1);
%     title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));
%     drawnow;
% end
%
%
%% Example 2 HAAR + ADABOOSTING%%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.negmax_size        = 1000;
% options.Npostrain          = 500;
% options.Nnegtrain          = 500;
% options.Npostest           = 250;
% options.Nnegtest           = 250;
% options.Nposboost          = 0;
% options.Nnegboost          = 50;
% options.boost_ite          = 3;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.standardize        = 1;
% options.preview            = 0;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.05;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.probaswitchIneg    = 0.9;
% options.posscalemin        = 0.4;
% options.posscalemax        = 1.5;
% options.negscalemin        = 0.5;
% options.negscalemax        = 1.1;
% options.addbias            = 1;
% options.max_detections     = 5000;
%
% options.typefeat           = 0;
% options.spyr               = [1 , 1 , 1 , 1 , 1; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
% options.scale              = [1];
% options.maptable           = 0;
% options.cs_opt             = 1;
% options.improvedLBP        = 0;
% options.rmextremebins      = 0;
% options.color              = 0;
% options.norm               = [0 , 0 , 2];
% options.clamp              = 0.2;
% options.n                  = 0;
% options.L                  = 1.2;
% options.kerneltype         = 0;
% options.numsubdiv          = 8;
% options.minexponent        = -20;
% options.maxexponent        = 8;
%
% options.dimsItraining      = [24 , 24];
% options.rect_param         = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
% options.usesingle          = 1;
% options.transpose          = 1;
%
% options.typeclassifier     = 0;
% options.s                  = 2;
% options.B                  = 1;
% options.c                  = 2;
% options.T                  = 50;
%
% options.num_threads        = -1;
% options.dimsIscan          = [24 , 24];
% options.scalingbox         = [2 , 1.4 , 1.8];
% options.mergingbox         = [1/2 , 1/2 , 1/3];
%
% [options , model]          = train_model(options);
%
% figure(1)
% plot(options.fpp , options.tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , options.perftest , options.auc_est))
% axis([-0.02 , 0.3 , 0.75 , 1.02])
%
% figure(2)
% plot(options.fxtest)
%
%
%% Example 3 MBLBP + ADABOOSTING %%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.negmax_size        = 1000;
% options.Npostrain          = 1000;
% options.Nnegtrain          = 1000;
% options.Npostest           = 500;
% options.Nnegtest           = 500;
% options.Nposboost          = 0;
% options.Nnegboost          = 100;
% options.boost_ite          = 3;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.standardize        = 1;
% options.preview            = 0;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.05;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.probaswitchIneg    = 0.9;
% options.posscalemin        = 0.4;
% options.posscalemax        = 1.5;
% options.negscalemin        = 0.5;
% options.negscalemax        = 1.1;
% options.addbias            = 1;
% options.max_detections     = 5000;
%
% options.typefeat           = 1;
% options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
% options.scale              = [1];
% options.maptable           = 0;
% options.cs_opt             = 1;
% options.improvedLBP        = 0;
% options.rmextremebins      = 0;
% options.color              = 0;
% options.norm               = [0 , 0 , 2];
% options.clamp              = 0.2;
% options.n                  = 0;
% options.L                  = 1.2;
% options.kerneltype         = 0;
% options.numsubdiv          = 8;
% options.minexponent        = -20;
% options.maxexponent        = 8;
%
% options.dimsItraining      = [24 , 24];
% options.usesingle          = 1;
% options.transpose          = 1;
%
% options.typeclassifier     = 0;
% options.s                  = 2;
% options.B                  = 1;
% options.c                  = 2;
% options.T                  = 50;
%
% options.num_threads        = -1;
% options.dimsIscan          = [24 , 24];
% options.scalingbox         = [2 , 1.4 , 1.8];
% options.mergingbox         = [1/2 , 1/2 , 1/3];
%
% [options , model]          = train_model(options);
%
% figure(1)
% plot(options.fpp , options.tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , options.perftest , options.auc_est))
% axis([-0.02 , 0.3 , 0.75 , 1.02])
%
% figure(2)
% plot(options.fxtest)
%
%
%% Example 4 HMBLGP + LSVM %%
%
% close all
% %options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
% %options.posext            = {'png'};
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.negmax_size        = 1000;
% options.Npostrain          = 10000;
% options.Nnegtrain          = 10000;
% options.Npostest           = 10000;
% options.Nnegtest           = 10000;
% options.Nposboost          = 50;
% options.Nnegboost          = 1000;
% options.boost_ite          = 5;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.standardize        = 1;
% options.preview            = 0;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.05;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.probaswitchIneg    = 0.9;
% options.posscalemin        = 0.4;
% options.posscalemax        = 1.5;
% options.negscalemin        = 0.5;
% options.negscalemax        = 1.1;
% options.addbias            = 1;
% options.max_detections     = 5000;
%
% options.typefeat           = 3;
% options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
% options.scale              = [1];
% options.maptable           = 0;
% options.cs_opt             = 1;
% options.improvedLGP        = 0;
% options.rmextremebins      = 0;
% options.color              = 0;
% options.norm               = [0 , 0 , 2];
% options.clamp              = 0.2;
% options.n                  = 0;
% options.L                  = 1.2;
% options.kerneltype         = 0;
% options.numsubdiv          = 8;
% options.minexponent        = -20;
% options.maxexponent        = 8;
%
% options.dimsItraining      = [24 , 24];
% options.rect_param         = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
% options.usesingle          = 1;
% options.transpose          = 0;
%
% options.typeclassifier     = 2;
% options.s                  = 2;
% options.B                  = 1;
% options.c                  = 2;
% options.T                  = 50;
%
% options.num_threads        = -1;
% options.dimsIscan          = [24 , 24];
% options.scalingbox         = [2 , 1.4 , 1.8];
% options.mergingbox         = [1/2 , 1/2 , 1/3];
%
% [options , model]          = train_model(options);
%
% figure(1)
% plot(options.fpp , options.tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , options.perftest , options.auc_est))
% axis([-0.02 , 0.3 , 0.75 , 1.02])
%
% figure(2)
% plot(model.w)
%
% figure(3)
% plot(options.fpp , options.tpp  , 'linewidth' , 2)
% axis([-0.02 , 1.02 , -0.02 , 1.02])
% %legend('Cascade' , 'MultiExit', 'Full', 'Location' , 'SouthEast')
% grid on
% title(sprintf('ROC for HMBLBP + Linear SVM'))
%
% figure(4)
% semilogy(1:options.boost_ite , options.pd_per_stage , 1:options.boost_ite , options.pfa_per_stage , 'linewidth' , 2)
% legend('P_d' , 'P_{fa}' , 'location' , 'southwest')
%
% aa                       = vcapg2(0,3);
% min_detect               = 2;%120;
% figure(5);set(1,'doublebuffer','on');
% while(1)
%     t1   = tic;
%     aa   = vcapg2(0,0);
%     pos  = detector_mlhmslbp_spyr(rgb2gray(aa) , model);
%     image(aa);
%     hold on
%     h    = plot_rectangle(pos(: , (pos(4 , :) >=min_detect)) , 'g' );
%     hold off
%     t2   = toc(t1);
%     title(sprintf('Fps = %6.3f      (Press CRTL+C to stop)' , 1/t2));
%     drawnow;
% end
%




if(nargin < 1)
    options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
    options.posext             = {'png'};
    options.negext             = {'jpg'};
    options.negmax_size        = 1000;
    options.standardize        = 1;
    options.preview            = 0;
    options.seed               = 5489;
    options.resetseed          = 1;
    options.Npostrain          = 10000;
    options.Nnegtrain          = 10000;
    options.Npostest           = 10000;
    options.Nnegtest           = 10000;
    options.Nposboost          = 50;
    options.Nnegboost          = 1000;
    options.boost_ite          = 5;
    options.probaflipIpos      = 0.5;
    options.probarotIpos       = 0.05;
    options.m_angle            = 0;
    options.sigma_angle        = 7^2;
    options.probaswitchIneg    = 0.9;
    options.posscalemin        = 0.25;
    options.posscalemax        = 1.75;
    options.negscalemin        = 0.7;
    options.negscalemax        = 3;
    
    options.max_detections     = 5000;
    options.num_threads        = -1;
    options.dimsIscan          = [24 , 24];
    options.scalingbox         = [2 , 1.4 , 1.8];
    options.mergingbox         = [1/2 , 1/2 , 1/3];
    
    
    options.typefeat           = 2;
    
    options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
    options.scale              = [1];
    options.maptable           = 0;
    options.cs_opt             = 1;
    options.improvedLBP        = 0;
    options.improvedLGP        = 0;
    options.rmextremebins      = 0;
    options.color              = 0;
    options.norm               = [0 , 0 , 2];
    options.clamp              = 0.2;
    options.n                  = 0;
    options.L                  = 1.2;
    options.kerneltype         = 0;
    options.numsubdiv          = 8;
    options.minexponent        = -20;
    options.maxexponent        = 8;
    
    options.dimsItraining      = [24 , 24];
    options.rect_param         = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
    options.usesingle          = 1;
    options.transpose          = 0;
    
    options.typeclassifier     = 2;
    
    options.addbias            = 1;
    options.s                  = 2;
    options.B                  = 1;
    options.c                  = 2;
    
    options.T                  = 50;
    
end


if(~any(strcmp(fieldnames(options) , 'positives_path')))
    options.positives_path      = fullfile(pwd , 'images' , 'train' , 'positives');
end
if(~any(strcmp(fieldnames(options) , 'negatives_path')))
    options.negatives_path      = fullfile(pwd , 'images' , 'train' , 'negatives');
end
if(~any(strcmp(fieldnames(options) , 'posext')))
    options.posext              = {'png'};
end
if(~any(strcmp(fieldnames(options) , 'negext')))
    options.negext              = {'jpg'};
end
if(~any(strcmp(fieldnames(options) , 'negmax_size')))
    options.negmax_size         = 1000;
end
if(~any(strcmp(fieldnames(options) , 'seed')))
    options.seed                = 5489;
end
if(~any(strcmp(fieldnames(options) , 'resetseed')))
    options.resetseed           = 1;
end
if(~any(strcmp(fieldnames(options) , 'standardize')))
    options.standardize         = 1;
end
if(~any(strcmp(fieldnames(options) , 'preview')))
    options.preview             = 0;
end
if(~any(strcmp(fieldnames(options) , 'Npostrain')))
    options.Npostrain           = 5000;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtrain')))
    options.Nnegtrain           = 10000;
end
if(~any(strcmp(fieldnames(options) , 'Npostest')))
    options.Npostest            = 5000;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtest')))
    options.Nnegtest            = 5000;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat            = 2;
end
if(~any(strcmp(fieldnames(options) , 'addbias')))
    options.addbias             = 1;
end
if(~any(strcmp(fieldnames(options) , 'Nposboost')))
    options.Nposboost           = 50;
end
if(~any(strcmp(fieldnames(options) , 'Nnegboost')))
    options.Nnegboost           = 1000;
end
if(~any(strcmp(fieldnames(options) , 'boost_ite')))
    options.boost_ite           = 10;
end
if(~any(strcmp(fieldnames(options) , 'probaswitchIneg')))
    options.probaswitchIneg     = 0.005;
end
if(~any(strcmp(fieldnames(options) , 'probaflipIpos')))
    options.probaflipIpos       = 0.5;
end
if(~any(strcmp(fieldnames(options) , 'probarotIpos')))
    options.probarotIpos        = 0.05;
end
if(~any(strcmp(fieldnames(options) , 'm_angle')))
    options.m_angle             = 0.0;
end
if(~any(strcmp(fieldnames(options) , 'sigma_angle')))
    options.sigma_angle         = 5^2;
end
if(~any(strcmp(fieldnames(options) , 'n')))
    options.n                   = 0;
end
if(~any(strcmp(fieldnames(options) , 'L')))
    options.L                   = 0.7;
end
if(~any(strcmp(fieldnames(options) , 'kerneltype')))
    options.kerneltype          = 0;
end
if(~any(strcmp(fieldnames(options) , 'numsubdiv')))
    options.numsubdiv           = 8;
end
if(~any(strcmp(fieldnames(options) , 'minexponent')))
    options.numsubdiv           = -20;
end
if(~any(strcmp(fieldnames(options) , 'maxexponent')))
    options.maxexponent         = 8;
end
if(~any(strcmp(fieldnames(options) , 's')))
    options.s                   = 2;
end
if(~any(strcmp(fieldnames(options) , 'B')))
    options.B                   = options.addbias;
end
if(~any(strcmp(fieldnames(options) , 'c')))
    options.c                   = 2;
end
if(~any(strcmp(fieldnames(options) , 'max_detections')))
    options.max_detections      = 5000;
end
if(~any(strcmp(fieldnames(options) , 'num_threads')))
    options.num_threads         = -1;
end
if(~any(strcmp(fieldnames(options) , 'dimsIscan')))
    options.dimsIscan           = [24 , 24];
end
if(~any(strcmp(fieldnames(options) , 'scalingbox')))
    options.scalingbox          = [2 , 1.4 , 1.8];
end
if(~any(strcmp(fieldnames(options) , 'mergingbox')))
    options.mergingbox          = [1/2 , 1/2 , 0.8];
end
if(~any(strcmp(fieldnames(options) , 'spyr')))
    options.spyr                = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
end
if(~any(strcmp(fieldnames(options) , 'scale')))
    options.scale               = 1;
end
if(~any(strcmp(fieldnames(options) , 'maptable')))
    options.maptable            = 0;
end
if(~any(strcmp(fieldnames(options) , 'cs_opt')))
    options.cs_opt              = 1;
end
if(~any(strcmp(fieldnames(options) , 'improvedLBP')))
    options.improvedLBP         = 0;
end
if(~any(strcmp(fieldnames(options) , 'improvedLGP')))
    options.improvedLGP         = 0;
end
if(~any(strcmp(fieldnames(options) , 'rmextremebins')))
    options.rmextremebins       = 0;
end
if(~any(strcmp(fieldnames(options) , 'color')))
    options.color               = 0;
end
if(~any(strcmp(fieldnames(options) , 'norm')))
    options.norm                = [0 , 0 , 2];
end
if(~any(strcmp(fieldnames(options) , 'clamp')))
    options.clamp               = 0.2;
end
if(~any(strcmp(fieldnames(options) , 'posscalemin')))
    options.posscalemin         = 0.25;
end
if(~any(strcmp(fieldnames(options) , 'posscalemax')))
    options.posscalemax         = 1.75;
end
if(~any(strcmp(fieldnames(options) , 'negscalemin')))
    options.posscalemin         = 0.7;
end
if(~any(strcmp(fieldnames(options) , 'negscalemax')))
    options.posscalemax         = 3;
end
if(~any(strcmp(fieldnames(options) , 'dimsItraining')))
    options.dimsItraining       = [24 , 24];
end
if(~any(strcmp(fieldnames(options) , 'transpose')))
    options.transpose           = 0;
end
if(~any(strcmp(fieldnames(options) , 'usesingle')))
    options.single              = 1;
end
if(~any(strcmp(fieldnames(options) , 'T')))
    options.T                   = 50;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat            = 2;
end

ny                              = options.dimsItraining(1);
nx                              = options.dimsItraining(2);
if( (options.typefeat == 0) && ~any(strcmp(fieldnames(options) , 'rect_param')))
    options.rect_param          = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2 ; 1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
end
if( (options.typefeat == 1) && ~any(strcmp(fieldnames(options) , 'map')))
    options.map                 = uint8(0:255);
end


if((options.typefeat == 0) && ~any(strcmp(fieldnames(options) , 'F')))
    options.F                 = haar_featlist(ny , nx , options.rect_param);
    options.indexF            = int32(0:size(options.F,2)-1);
else
    options.F                 = mblbp_featlist(ny , nx);
    options.indexF            = int32(0:size(options.F,2)-1);
end

if((options.usesingle == 1) && (options.n > 0))
    options.n                  = 0;
end

if(options.typeclassifier == 0)
    options.weaklearner = 2;
elseif(options.typeclassifier == 1)
    options.weaklearner = 0;
end

%% reset seed eventually %%

if(options.resetseed)
    RandStream.setDefaultStream(RandStream.create('mt19937ar','seed',options.seed));
end
options.resetseed                                                     = 0;

%% generate Train & Test features
[options.Xtrain , options.ytrain , options.Xtest , options.ytest]     = generate_face_features(options);

% indp                                                                  = find(options.ytrain == 1);
% indn                                                                  = find(options.ytrain ==-1);
% indneg                                                                = options.Npostrain+1:options.Npostrain+options.Nnegtrain;
% indnegboost                                                           = options.Npostrain+1:options.Npostrain+options.Nnegtrain+options.Nnegboost;

if((options.n > 0))
    fprintf('\nComputing homogeneous kernel approximation table and approximated data\n');
    drawnow
    options.homtable                                                  = homkertable(options);
    options.Xtrain                                                    = homkermap(options.Xtrain , options);
end

fprintf('\nTraining (P+N) sets\n');
drawnow

if(options.typeclassifier == 0)
    model                                                             = haar_adaboost_binary_train_cascade_memory(options.Xtrain , int8(options.ytrain) , options);
    options.param                                                     = model;
    options.cascade_type                                              = 0;
    options.cascade                                                   = [size(options.param , 2) ; 0];
    [options.ytrain_est , options.fxtrain]                            = haar_adaboost_binary_predict_cascade_memory(options.Xtrain , options);
elseif(options.typeclassifier == 1)
    model                                                             = haar_gentleboost_binary_train_cascade_memory(options.Xtrain , int8(options.ytrain) , options);
    options.param                                                     = model;
    options.cascade_type                                              = 0;
    options.cascade                                                   = [size(options.param , 2) ; 0];
    [options.ytrain_est , options.fxtrain]                            = haar_gentleboost_binary_predict_cascade_memory(options.Xtrain , options);
elseif(options.typeclassifier == 2)
    model                                                             = train_dense(options.ytrain' , options.Xtrain , sprintf('-s %d -B %d -c %d' , options.s , options.B , options.c) , 'col');
    options.w                                                         = model.w;
    options.fxtrain                                                   = options.w(1:end-1)*options.Xtrain + options.w(end);
    if(options.addbias)
        options.fxtrain                                               = options.fxtrain + options.w(end);
    end
    options.ytrain_est                                                = sign(options.fxtrain);
end

options.m                                                             = 1;
options.pd_per_stage                                                  = zeros(1 , options.boost_ite);
options.pfa_per_stage                                                 = zeros(1 , options.boost_ite);

while (options.m <= options.boost_ite)
    
    [Xnd , fxnd , pd_current]                                         = generate_nd_features(options);
    options.pd_per_stage(options.m)                                   = pd_current;
    
    [Xfa , fxfa , pfa_current]                                        = generate_fa_features(options);
    options.pfa_per_stage(options.m)                                  = pfa_current;
    
    %     [valfx , indfx]                                                   = sort([options.fxtrain(indn) , fxfa] , 'descend');
    %     Xtemp                                                             = [options.Xtrain  , Xfa];
    %     uindfx                                                            = unique(indfx);
    %     options.Xtrain(: , indneg)                                        = Xtemp(: , indnegboost(uindfx(1:options.Nnegtrain)));
    
    if(options.n > 0)
        fprintf('\nComputing homogeneous kernel approximation table and approximated data\n');
        drawnow
        Xnd                                                           = homkermap(Xnd , options);
        Xfa                                                           = homkermap(Xfa , options);
    end
    if((options.transpose) && (options.typefeat < 2))
        options.Xtrain                                                = [options.Xtrain ; Xnd ; Xfa];
    else
        options.Xtrain                                                = [options.Xtrain , Xnd , Xfa];
    end
    options.ytrain                                                    = [options.ytrain , ones(1 , options.Nposboost) , -1*ones(1 , options.Nnegboost)];
    
    fprintf('\nTraining (P+dP+N+dN) sets, bootstrap stage m = %d\n' , options.m);
    drawnow
    
    if(options.typeclassifier == 0)
        model                                                         = haar_adaboost_binary_train_cascade_memory(options.Xtrain , int8(options.ytrain) , options);
        options.param                                                 = model;
        options.cascade_type                                          = 0;
        options.cascade                                               = [size(options.param , 2) ; 0];
        [options.fxtrain , options.fxtrain]                           = haar_adaboost_binary_predict_cascade_memory(options.Xtrain , options);
        options.fxtrain                                               = double(options.fxtrain);
    elseif(options.typeclassifier == 1)
        model                                                         = haar_gentleboost_binary_train_cascade_memory(options.Xtrain , int8(options.ytrain) , options);
        options.param                                                 = model;
        options.cascade_type                                          = 0;
        options.cascade                                               = [size(options.param , 2) ; 0];
        [options.fxtrain , options.fxtrain]                           = haar_gentleboost_binary_predict_cascade_memory(options.Xtrain , model);
        options.fxtrain                                               = double(options.fxtrain);
    elseif(options.typeclassifier == 2)
        model                                                         = train_dense(options.ytrain' , options.Xtrain , sprintf('-s %d -B %d -c %d' , options.s , options.B , options.c) , 'col');
        options.w                                                     = model.w;
        options.fxtrain                                               = options.w(1:end-1)*options.Xtrain + options.w(end);
        if(options.addbias)
            options.fxtrain                                           = options.fxtrain + options.w(end);
        end
    end
    
    options.m                                                         = options.m + 1;
    %    options.m
end

if(options.n > 0)
    options.Xtest                                                     = homkermap(options.Xtest , options);
end

indp                                                                  = find(options.ytest == 1);
indn                                                                  = find(options.ytest ==-1);

if(options.typeclassifier == 0)
    [options.ytest_est , options.fxtest]                              = haar_adaboost_binary_predict_cascade_memory(options.Xtest , options);
    options.ytest_est                                                 = double(options.ytest_est);
elseif(options.typeclassifier == 1)
    [options.ytest_est , options.fxtest]                              = haar_gentleboost_binary_predict_cascade_memory(options.Xtest , options);
    options.ytest_est                                                 = double(options.ytest_est);
elseif(options.typeclassifier == 2)
    options.fxtest                                                    = options.w(1:end-1)*options.Xtest;
    if(options.addbias)
        options.fxtest                                                = options.fxtest + options.w(end);
    end
    if(model.Label(1)==-1)
        options.fxtest                                                = -options.fxtest;
    end
    options.ytest_est                                                 = sign(options.fxtest);
end
options.tp                                                            = sum(options.ytest_est(indp) == options.ytest(indp))/length(indp);
options.fp                                                            = 1 - sum(options.ytest_est(indn) == options.ytest(indn))/length(indn);
options.perftest                                                      = sum(options.ytest_est == options.ytest)/length(options.ytest);
[options.tpp , options.fpp]                                           = basicroc(options.ytest , options.fxtest);
options.auc_est                                                       = auroc(options.tpp', options.fpp');

clear model
if(options.typefeat == 0)
    model.param                                                       = options.param;
    model.weaklearner                                                 = options.weaklearner;
    model.dimsItraining                                               = options.dimsItraining;
    model.rect_param                                                  = options.rect_param;
    model.F                                                           = options.F;
    model.cascade_type                                                = 0;
    model.cascade                                                     = [size(options.param , 2) ; 0];
elseif(options.typefeat == 1)
    model.param                                                       = options.param;
    model.weaklearner                                                 = options.weaklearner;
    model.dimsItraining                                               = options.dimsItraining;
    model.F                                                           = options.F;
    model.map                                                         = options.map;
    model.cascade_type                                                = 0;
    model.cascade                                                     = [size(options.param , 2) ; 0];
elseif(options.typefeat == 2)
    model.w                                                           = options.w;
    model.spyr                                                        = options.spyr;
    model.nH                                                          = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
    model.scale                                                       = options.scale;
    model.maptable                                                    = options.maptable;
    model.cs_opt                                                      = options.cs_opt;
    model.improvedLBP                                                 = options.improvedLBP;
    model.rmextremebins                                               = options.rmextremebins;
    model.norm                                                        = options.norm;
    model.clamp                                                       = options.clamp;
    model.addbias                                                     = options.addbias;
    model.n                                                           = options.n;
    model.L                                                           = options.L;
    model.kerneltype                                                  = options.kerneltype;
    model.numsubdiv                                                   = options.numsubdiv;
    model.minexponent                                                 = options.minexponent;
    model.maxexponent                                                 = options.maxexponent;
    model.dimsIscan                                                   = options.dimsIscan;
elseif(options.typefeat == 3)
    model.w                                                           = options.w;
    model.spyr                                                        = options.spyr;
    model.nH                                                          = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
    model.scale                                                       = options.scale;
    model.maptable                                                    = options.maptable;
    model.cs_opt                                                      = options.cs_opt;
    model.improvedLGP                                                 = options.improvedLGP;
    model.rmextremebins                                               = options.rmextremebins;
    model.norm                                                        = options.norm;
    model.clamp                                                       = options.clamp;
    model.addbias                                                     = options.addbias;
    model.n                                                           = options.n;
    model.L                                                           = options.L;
    model.kerneltype                                                  = options.kerneltype;
    model.numsubdiv                                                   = options.numsubdiv;
    model.minexponent                                                 = options.minexponent;
    model.maxexponent                                                 = options.maxexponent;
    model.dimsIscan                                                   = options.dimsIscan;
end
model.scalingbox                                                      = options.scalingbox;
model.mergingbox                                                      = options.mergingbox;
model.postprocessing                                                  = 1;
model.max_detections                                                  = options.max_detections;
model.num_threads                                                     = options.num_threads;
