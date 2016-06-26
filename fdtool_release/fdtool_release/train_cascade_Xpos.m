function [options , model] = train_cascade_Xpos(Xpos , options)
%
%  Train model for Haar/MBLBP features + Adaboosting/Gentleboosting 
%  + Conventional/Asymetric cascade
%  Positives examples are stacked in a 3D tensor matrix Xpos
%
%  Usage
%  ------
%
%  [options , [model]] = train_cascade_Xpos(Xpos , options)
%
%  Inputs
%  -------
%
%  Xpos              Positives database (Ny x Nx x Npostotal)
%
%
%  options           Strucure
%
%                    negatives_path  Path contening Negatives pictures in jpeg format
%                    negext          Extention files for negatives pictures
%                    negmax_size     Maximum side of each negative image (default = 1000)
%                    seed            Seed value for random generator in order to generate the same positive/negative sets
%                    resetseed       Reset generator with given seed (default resetseed = 1)  
%                    preview         Preview current example (default preview = 0)
%                    standardize     Standardize images (default standardize = 1)
%                    Npostrain       Number of desired positives examples training each stage (Npostrain+Npostest <= Npostotal)
%                    Nnegtrain       Number of desired negatives examples for training each stage. Extracted by bilinear
%                                    interpolation from the negatives database
%                    Npostest        Number of desired positives examples for training each stage (Npostrain+Npostest <= Npostotal)
%                    Nnegtest        Number of desired negatives examples for training each stage. Extracted by bilinear
%                                    interpolation from the negatives database
%                    cascade_type    Type of cascade: 0 <=> conventional, cascade, 1 <=> multiexit cascade
%                    maxstage        Maximum number of stages (default maxstage = 20)
%                    maxwl_perstage  Maximum number of weaklearner for each stage (default maxweaklearner = 200)
%                    maxK            Number of new weaklearner added without a change in alpha_m or beta_m (default maxK =5)
%                    alpha0          Desired False Acceptance Rate for each stage (default alpha0 = 0.5)
%                                    alpha0 can be a (1 x options.maxstage) vector
%                    beta0           Desired False Rejection Rate for each stage (default beta0 = 0.01)
%                                    beta0 can be a (1 x options.maxstage) vector
%                    typefeat        Type of Features to use: 0 <=> Haar features, 1 <=> MBLPB features (default typefeat = 0)
%                    algoboost       Type of boosting algorithm, 0 = gentleboost, 1 = Adaboost, 2  = Fastadaboost 
%                                    with Haar features (if options.typefeat=0)), 3 = Haargentleboost requiering a lot of memory
%                                    4 = Haaradaboost requiering a lot of memory(default algoboost = 2)
%                    transpose       Transpose input to speed up and save memory (for MBLBP and Haar Adaboosting, algoboost = 1,3)
%                                    (default transpose = 0)
%                    rect_param      Pattern matrix (10 x nR) (requiered if typefeat = 0)(default 2 Patterns)
%                    map             Mapping MBLBP features (requiered if typefeat = 1) (default map = (0:255))
%                    probaflipIpos   Probability to flip Positive examples (default probaflipIpos = 0.5)
%                    probarotIpos    Probability to rotate Positives examples with an angle~N(m_angle,sigma_angle) (default probarotIpos = 0.01)
%                    m_angle         Mean rotation angle value in degree (default mangle = 0)
%                    sigma_angle     variance of the rotation angle value (default sigma_angle = 5^2)
%                    probaswitchIneg Probability to swith from another picture in the negatives database (default probaswitchIneg = 0.005)
%                    usefa           Use previous False alarms to construct negatives (default usefa = 0)
%                    scalemin        Minimum scaling factor to apply on negatives patch subwindows (default scalemin = 1)
%                    scalemax        Maximum scaling factor to apply on negatives patch subwindows (default scalemax = 5)
%                    num_threads     Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)
%
%  Outputs
%  -------
%
%  options          Input options + trained cascade
%                   param            Full cascade parameters for each trained weaklearner (4 x nb_weaklearner)
%                   cascade          Cascade matrix (2 x nb_stage) where cascade(1 , :) denotes the number of weaklearner per stage and cascade(2 , :) the threshold to apply for each stage
%                   stats            Basic statictics for each stages (4 x nb_stage)
%                   alpha_perstage   Estimated False Acceptance Rate for each stage (1 x nb_stage)
%                   beta_perstage    Estimated False Rejection Rate for each stage (1 x nb_stage)
%
%  model            Model structure compatible with detector input (see detector_haar or detector_mblbp)
%
%
%  Example 1 : Haar feature + Fastadaboost + conventional cascade
%  --------------------------------------------------------------
%
%  load viola_24x24
%  Xpos                       = X(: , : , find(y == 1));
%  load jensen_24x24
%  Xpos                       = cat(3 , Xpos , X(: , : , find(y == 1)));
%  options                    = load('haar_dico_2.mat');
%  options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
%  options.negext             = {'jpg'};
%  options.negmax_size        = 1000;
%  options.standardize        = 1;
%  options.seed               = 5489;
%  options.resetseed          = 1;
%  options.preview            = 0;
%  options.Npostrain          = 5000;
%  options.Nnegtrain          = 5000;
%  options.Npostest           = 2000;
%  options.Nnegtest           = 2000;
%  options.cascade_type       = 0;
%  options.alpha0             = 0.8;   %false acceptance rate
%  options.beta0              = 0.01;  %false rejection rate
%  options.maxwl_perstage     = 30;
%  options.maxstage           = 12;
%  options.typefeat           = 0;   %0 = Haar, 1 = MBLBP
%  options.algoboost          = 2;   %0 = Gentleboost, 1 = Adaboost, 2 = Fastadaboost (if  options.typefeat =0)
%  options.fine_threshold     = 1;
%  options.maxK               = 30;
%  options.probaflipIpos      = 0.5;
%  options.probarotIpos       = 0.05;
%  options.m_angle            = 0;
%  options.sigma_angle        = 5^2;
%  options.probaswitchIneg    = 0.05;
%  options.usefa              = 1;
%  options.scalemin           = 1;
%  options.scalemax           = 6;
%  options.num_threads        = -1;
%
%  [options , model]          = train_cascade_Xpos(Xpos , options);
%
%  Pfa_cascade                = prod(options.alpha0);
%  Pd_cascade                 = prod(1-options.beta0);
%  Pfa_cascade_est            = prod(options.alphaperstage);
%  Pd_cascade_est             = prod(1-options.betaperstage);
%
%  I                          = (rgb2gray(imread('class57.jpg')));
%  tic,[D , stat]             = detector_haar(I , model);,toc
% 
%  figure, imshow(I);
%  hold on;
%  h = plot_rectangle(D , 'r');
%  hold off
%  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))
%
%
%  Example 2 : Haar feature + Fastadaboost + Multiexit cascade
%  --------------------------------------------------------------
%
%  load viola_24x24
%  Xpos                       = X(: , : , find(y == 1));
%  load jensen_24x24
%  Xpos                       = cat(3 , Xpos , X(: , : , find(y == 1)));
%  options                    = load('haar_dico_2.mat');
%  options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
%  options.negext             = {'jpg'};
%  options.negmax_size        = 1000;
%  options.standardize        = 1;
%  options.seed               = 5489;
%  options.resetseed          = 1;
%  options.preview            = 0;
%  options.Npostrain          = 6000;
%  options.Nnegtrain          = 6000;
%  options.Npostest           = 0;
%  options.Nnegtest           = 0;
%  options.cascade_type       = 1;
%  options.alpha0             = 0.80;   %false acceptance rate
%  options.beta0              = 0.01;   %false rejection rate
%  options.maxwl_perstage     = 200;
%  options.maxstage           = 24;
%  options.typefeat           = 0;   %0 = Haar, 1 = MBLBP
%  options.algoboost          = 2;   %0 = Gentleboost, 1 = Adaboost, 2 = Fastadaboost (if  options.typefeat =0)
%  options.fine_threshold     = 1;
%  options.maxK               = 200;
%  options.probaflipIpos      = 0.5;
%  options.probarotIpos       = 0.05;
%  options.m_angle            = 0;
%  options.sigma_angle        = 5^2;
%  options.probaswitchIneg    = 0.05;
%  options.usefa              = 0;
%  options.scalemin           = 1;
%  options.scalemax           = 6;
%  options.num_threads        = -1;
%
%  [options , model]          = train_cascade_Xpos(Xpos , options);
%% save fast_haar_ada_model  model
%
%  Pfa_cascade                = prod(options.alpha0);
%  Pd_cascade                 = prod(1-options.beta0);
%  Pfa_cascade_est            = prod(options.alphaperstage);
%  Pd_cascade_est             = prod(1-options.betaperstage);
%
%  I                          = (rgb2gray(imread('class57.jpg')));
%  tic,[D , stat]             = detector_haar(I , model);,toc
% 
%  min_detect                 = 20;
%  model.scalingbox           = [1.5 , 1.3 , 2];
%  figure, imshow(I);
%  hold on;
%  h = plot_rectangle(D(: , (D(4 , :) >=min_detect)) , 'g');
%  hold off
%  title(sprintf('nF = %d, Detect = %5.4f%%, Non-Detect = %5.4f%%' , size(model.param , 2) , 100*stat(1)/sum(stat) , 100*stat(2)/sum(stat)))
%
%  Example 3 : Haar features pre-computed + Gentleboosting (memory) + Multiexit cascade
%  -------------------------------------------------------------------------------------
%
%  load viola_24x24
%  Xpos                       = X(: , : , find(y == 1));
%  load jensen_24x24
%  Xpos                       = cat(3 , Xpos , X(: , : , find(y == 1)));
%  options                    = load('haar_dico_2.mat');
%  options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
%  options.negext             = {'jpg'};
%  options.negmax_size        = 1000;
%  options.standardize        = 1;
%  options.seed               = 5489;
%  options.resetseed          = 1;
%  options.preview            = 0;
%  options.Npostrain          = 500;
%  options.Nnegtrain          = 500;
%  options.Npostest           = 0;
%  options.Nnegtest           = 0;
%  options.cascade_type       = 1;
%  options.alpha0             = 0.80;   %false acceptance rate
%  options.beta0              = 0.01;   %false rejection rate
%  options.maxwl_perstage     = 200;
%  options.maxstage           = 24;
%  options.typefeat           = 0;   %0 = Haar, 1 = MBLBP
%  options.algoboost          = 3;   %0 = Gentleboost, 1 = Adaboost, 2 = Fastadaboost (if  options.typefeat =0)
%  options.transpose          = 1;
%  options.maxK               = 200;
%  options.probaflipIpos      = 0.5;
%  options.probarotIpos       = 0.05;
%  options.m_angle            = 0;
%  options.sigma_angle        = 5^2;
%  options.probaswitchIneg    = 0.05;
%  options.usefa              = 0;
%  options.scalemin           = 1;
%  options.scalemax           = 6;
%  options.num_threads        = -1;
%
%  [options , model]          = train_cascade_Xpos(Xpos  , options);
%% save fast_haar_ada_model  model
%
%  Pfa_cascade                = prod(options.alpha0);
%  Pd_cascade                 = prod(1-options.beta0);
%  Pfa_cascade_est            = prod(options.alphaperstage);
%  Pd_cascade_est             = prod(1-options.betaperstage);
%
%  Example 4 : MBLBP features pre-computed + Gentleboosting (memory) + Multiexit cascade
%  -------------------------------------------------------------------------------------
%
%  load viola_24x24
%  Xpos                       = X(: , : , find(y == 1));
%  load jensen_24x24
%  Xpos                       = cat(3 , Xpos , X(: , : , find(y == 1)));
%  options                    = load('haar_dico_2.mat');
%  options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
%  options.negext             = {'jpg'};
%  options.negmax_size        = 1000;
%  options.standardize        = 1;
%  options.seed               = 5489;
%  options.resetseed          = 1;
%  options.preview            = 0;
%  options.Npostrain          = 5000;
%  options.Nnegtrain          = 5000;
%  options.Npostest           = 0;
%  options.Nnegtest           = 0;
%  options.cascade_type       = 1;
%  options.alpha0             = 0.80;   %false acceptance rate
%  options.beta0              = 0.01;   %false rejection rate
%  options.maxwl_perstage     = 20;
%  options.maxstage           = 12;
%  options.typefeat           = 1;   %0 = Haar, 1 = MBLBP
%  options.algoboost          = 0;   %0 = Gentleboost, 1 = Adaboost, 2 = Fastadaboost (if  options.typefeat =0)
%  options.transpose          = 1;
%  options.maxK               = 20;
%  options.probaflipIpos      = 0.5;
%  options.probarotIpos       = 0.05;
%  options.m_angle            = 0;
%  options.sigma_angle        = 5^2;
%  options.probaswitchIneg    = 0.05;
%  options.usefa              = 0;
%  options.scalemin           = 1;
%  options.scalemax           = 6;
%  options.num_threads        = -1;
%
%  [options , model]          = train_cascade_Xpos(Xpos , options);
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%
%  Ref :   M-T. Pham and all, "Detection with multi-exit asymetric boosting", CVPR'08
%  ------
%

if(nargin < 1)
    error('Positives samples matrix (Ny x Nx x Npostotal) is requiered');
end

[ny , nx , Npostotal]         = size(Xpos);

if(nargin < 2)
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
    options.negext             = {'jpg'};
    options.negmax_size        = 1000;
    options.standardize        = 1;    
    options.seed               = 5489;
    options.resetseed          = 1;
    options.preview            = 0;    
    options.Npostrain          = 5000;
    options.Nnegtrain          = 5000;
    options.Npostest           = 1000;
    options.Nnegtest           = 1000;
    options.cascade_type       = 0;
    options.alpha0             = 0.5;
    options.beta0              = 0.01;
    options.maxwl_perstage     = 30;
    options.maxstage           = 15;
    options.maxK               = 10;
    options.typefeat           = 0;   %0 = Haar, 1 = MBLBP
    options.algoboost          = 2;   %0 = Gentleboost, 1 = Adaboost, 2 = Fastadaboost (if  options.typefeat =0)
    options.fine_threshold     = 1;
    options.transpose          = 0;
    options.usesingle          = 1;
    options.map                = uint8(0:255);
    options.resetseed          = 1;
    options.probaflipIpos      = 0.5;
    options.probarotIpos       = 0.05;
    options.m_angle            = 0;
    options.sigma_angle        = 5^2;
    options.probaswitchIneg    = 0.005;
    options.usefa              = 0;
    options.scalemin           = 1;
    options.scalemax           = 5;
    options.num_threads        = -1;
end

if(~any(strcmp(fieldnames(options) , 'negatives_path')))
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
end
if(~any(strcmp(fieldnames(options) , 'negext')))
    options.negext             = {'jpg'};
end
if(~any(strcmp(fieldnames(options) , 'negmax_size')))
    options.negmax_size        = 1000;
end
if(~any(strcmp(fieldnames(options) , 'seed')))
    options.seed                = 5489;
end
if(~any(strcmp(fieldnames(options) , 'resetseed')))
    options.resetseed            = 1;
end
if(~any(strcmp(fieldnames(options) , 'standardize')))
    options.standardize         = 1;
end
if(~any(strcmp(fieldnames(options) , 'preview')))
    options.preview             = 0;
end
if(~any(strcmp(fieldnames(options) , 'Npostrain')))
    options.Npostrain          = round(Npostotal)/3;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtrain')))
    options.Nnegtrain          = options.Npostrain;
end
if(~any(strcmp(fieldnames(options) , 'Npostest')))
    options.Npostest           = round(Npostotal)/6;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtest')))
    options.Nnegtest           = options.Npostest;
end
if(~any(strcmp(fieldnames(options) , 'cascade_type')))
    options.cascade_type       = 0;
end
if(~any(strcmp(fieldnames(options) , 'alpha0')))
    options.alpha0             = 0.5;
end
if(~any(strcmp(fieldnames(options) , 'beta0')))
    options.beta0              = 0.01;
end
if(~any(strcmp(fieldnames(options) , 'maxwl_perstage')))
    options.maxwl_perstage     = 20;
end
if(~any(strcmp(fieldnames(options) , 'maxstage')))
    options.maxstage           = 10;
end
if(~any(strcmp(fieldnames(options) , 'maxK')))
    options.maxK               = 5;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat           = 0;
end
if(~any(strcmp(fieldnames(options) , 'algoboost')))
    options.algoboost          = 2;
end
if(~any(strcmp(fieldnames(options) , 'fine_threshold')))
    options.fine_threshold     = 1;
end
if(~any(strcmp(fieldnames(options) , 'transpose')))
    options.transpose          = 0;
end
if(~any(strcmp(fieldnames(options) , 'usesingle')))
    options.single             = 1;
end
if(~any(strcmp(fieldnames(options) , 'map')))
    options.map                = uint8(0:255);
end
if(~any(strcmp(fieldnames(options) , 'probaswitchIneg')))
    options.probaswitchIneg    = 0.005;
end
if(~any(strcmp(fieldnames(options) , 'probaflipIpos')))
    options.probaflipIpos      = 0.5;
end
if(~any(strcmp(fieldnames(options) , 'probarotIpos')))
    options.probarotIpos       = 0.05;
end
if(~any(strcmp(fieldnames(options) , 'm_angle')))
    options.m_angle            = 0.0;
end
if(~any(strcmp(fieldnames(options) , 'sigma_angle')))
    options.sigma_angle        = 5^2;
end
if(~any(strcmp(fieldnames(options) , 'usefa')))
    options.usefa              = 0;
end
if(~any(strcmp(fieldnames(options) , 'scalemin')))
    options.scalemin           = 1;
end
if(~any(strcmp(fieldnames(options) , 'scalemax')))
    options.scalemax           = 5;
end

%% reset seed eventually %%
if(options.resetseed)
    RandStream.setDefaultStream(RandStream.create('mt19937ar','seed',options.seed));
end

dirneg                         = [];
for i = 1:length(options.negext)
    dirneg                     = [dirneg ; dir(fullfile(options.negatives_path , ['*.' options.negext{i}]))] ;
end
if(length(dirneg) < 1)
    error('Negatives directory is empty')
end
if(options.cascade_type == 0)
    if( options.Npostest == 0);
        options.Npostest          = 1000;
    end
    if( options.Nnegtest == 0);
        options.Nnegtest          = 1000;
    end
elseif(options.cascade_type == 1)
    %% Force empty test data since they are not used %%
    options.usefa             = 0;
%     options.Npostest          = 0;
%     options.Nnegtest          = 0;
end

if ((options.algoboost == 0) || (options.algoboost == 3)) %gentelboost
    options.weaklearner       = 0;
else
    options.weaklearner       = 2;
end
if(options.algoboost > 2)
    options.usesingle         = 1;
end
if( (options.typefeat == 0) && ~any(strcmp(fieldnames(options) , 'rect_param')))
    options.rect_param        = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
end
if( (options.typefeat == 1) && ~any(strcmp(fieldnames(options) , 'map')))
    options.map               = uint8(0:255);
end
if((options.algoboost == 2) && (options.typefeat ~= 0))
    disp('FastAdaboosting is only available for Haar Features, switching to normal Adaboosting');
    options.algoboost         = 1;
end
if(numel(options.alpha0) == 1)
    options.alpha0 = options.alpha0(: , ones(1 , options.maxstage));
end
if(numel(options.beta0) == 1)
    options.beta0 = options.beta0(: , ones(1 , options.maxstage));
end

options.param             = zeros(4 , 0);
options.cascade           = zeros(2 , 0);
options.stat              = zeros(7 , 0);
options.thresholdperstage = zeros(1 , 0);
options.alphaperstage     = zeros(1 , 0);
options.betaperstage      = zeros(1 , 0);
options.dimsItraining     = [ny , nx];
options.ny                = ny;
options.nx                = nx;

if(options.typefeat == 0)
    options.F                 = haar_featlist(ny , nx , options.rect_param);
    if(options.algoboost == 2)
        options.G             = haar_matG(ny , nx , options.rect_param);
        options.indexF        = int32(0:size(options.G,2)-1);
    else
        options.indexF        = int32(0:size(options.F,2)-1);
    end
else
    options.F                 = mblbp_featlist(ny , nx);
    options.indexF            = int32(0:size(options.F,2)-1);
end

%%
ytrain                        = [ones(1 , options.Npostrain , 'int8') , -ones(1 , options.Nnegtrain , 'int8')];
ytest                         = [ones(1 , options.Npostest , 'int8')  , -ones(1 , options.Nnegtest , 'int8')];

options.lambda                = options.alpha0/options.beta0;
options.std_angle             = sqrt(options.sigma_angle);
options.Npos                  = options.Npostrain + options.Npostest;
options.Nneg                  = options.Nnegtrain + options.Nnegtest;
options.Ntrain                = options.Npostrain + options.Nnegtrain;
options.Ntest                 = options.Npostest  + options.Nnegtest;
options.m                     = 0;
diffnodes                     = 0;

nb_stage                      = 1;
Xfa                           = zeros(ny , nx , 0);

fprintf('\n--------------- > pd_cascade_theo(%d) = %5.4f\n'  , options.maxstage ,  prod(1-options.beta0));
fprintf('--------------- > pfa_cascade_theo(%d) = %5.4f\n' , options.maxstage ,  prod(options.alpha0));

%% Main loop %%
while((diffnodes < options.maxwl_perstage) && (nb_stage <= options.maxstage))
    [Xtrain , Xtest , stat]       = generate_data_cascade_Xpos(Xpos , Xfa  , options);
    [options , Xfa]               = train_stage_cascade(Xtrain , ytrain , Xtest , ytest , options);
    options.stat(: , nb_stage)    = [stat(:) ; options.betaperstage(nb_stage) ; options.alphaperstage(nb_stage)];
    diffnodes                     = options.m(end);
    options.pfa_cascade(nb_stage) = prod(options.alphaperstage);
    options.pd_cascade(nb_stage)  = prod(1-options.betaperstage);
    fprintf('--------------- > pd_cascade(%d) = %5.4f\n'  , nb_stage ,  options.pd_cascade(nb_stage));
    fprintf('--------------- > pfa_cascade(%d) = %5.4f\n' , nb_stage ,  options.pfa_cascade(nb_stage));
    nb_stage                      = nb_stage + 1;
    drawnow
end

%%
if(diffnodes == options.maxwl_perstage)
    options.param             = options.param(: , end-options.maxwl_perstage);
    options.m(end)            = [];
end
if(nargout == 2)
    model.weaklearner         = options.weaklearner;
    model.param               = options.param;
    model.dimsItraining       = [ny , nx];
    model.cascade_type        = options.cascade_type;
    model.cascade             = options.cascade;
    model.scalingbox          = [2 , 1.35 , 1.75];
    model.mergingbox          = [1/2 , 1/2 , 0.8];    
    model.postprocessing      = 2;
    model.max_detections      = 1000;
    model.num_threads         = -1;
    if(options.typefeat == 0)
        model.F               = options.F;
        model.rect_param      = options.rect_param;
    elseif (options.typefeat == 1)
        model.F               = options.F;
        model.map             = options.map;
    end
end