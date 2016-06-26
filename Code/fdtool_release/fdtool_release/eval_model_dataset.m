function  [fx , y]     = eval_model_dataset(options , model)

%
%  Evaluate trained model on a set of extracted Positives and Negatives pictures
%  from positves and negatives folder respectively
%
%  Usage
%  ------
%
%  [fx , y]     = eval_model_dataset(options , model);
%
%  Inputs
%  -------
%  options          Options struture
%
%                   positives_path   Path from positives images are loaded for generating positives examples
%                   negatives_path   Path from negative images are loaded for generating negative examples
%                   posext           Positives extension files
%                   negext           Negatives extension files
%                   negmax_size      Maximum side of each negative image (default = 1000)
%                   standardize      Standardize images (default standardize = 1)
%                   seed             Seed value for random generator in order to generate the same positive sets
%                   resetseed        Reset generator with given seed(default resetseed = 1)  
%                   preview          Preview current example (default preview = 0)
%                   Npos             Number of desired positives examples
%                   Nneg             Number of desired negatives examples. Extracted by bilinear
%                                    interpolation from the negatives database
%                   probaflipIpos    Probability to flip Positive examples (default probaflipIpos = 0.5)
%                   probarotIpos     Probability to rotate Positives examples with an angle~N(m_angle,sigma_angle) (default probarotIpos = 0.01)
%                   m_angle          Mean rotation angle value in degree (default mangle = 0)
%                   sigma_angle      variance of the rotation angle value (default sigma_angle = 5^2)
%                   probaswitchIneg  Probability to swith from another picture in the negatives database (default probaswitchIneg = 0.005)
%                   posscalemin      Minimum scaling factor to apply on positives patch subwindows (default scalemin = 0.25)
%                   posscalemax      Maximum scaling factor to apply on positives patch subwindows (default scalemax = 2)
%                   negscalemin      Minimum scaling factor to apply on negatives patch subwindows (default scalemin = 1)
%                   negscalemax      Maximum scaling factor to apply on negatives patch subwindows (default scalemax = 5)
%                   featype          Type of features (featype: 0 <=> Haar, 1 <=> MBLBP, 3 <=> Histogram of MBLBP)
%
%   model                            Model structure of the detector
%                                    Haar/MBLBP/HMBLBP model accepted
%
%
%  Outputs
%  -------
%
%  fx               Predicted value for each extracted picts (1 x (options.Npos+options.Nneg))
%  y                True label (1 x (options.Npos+options.Nneg)) with yi=+1 for faces, yi=-1 else
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 02/25/2011
%
% Example 1  With HMSLBP_spyr
% ---------
%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'test' , 'negatives');
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.seed               = 5489;
% options.resetseed          = 1;
% options.standardize        = 1;
% options.preview            = 0;
% options.Npos               = 3000;
% options.Nneg               = 5000;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.00;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.probaswitchIneg    = 0.5;
% options.posscalemin        = 1;
% options.posscalemax        = 1;
% options.negscalemin        = 0.7;
% options.negscalemax        = 3;
% options.typefeat           = 3;
%
% load model_hmblbp_R4.mat
%
% [fx , y]                   = eval_model_dataset(options , model);
% yest                       = sign(fx);
% accuracy                   = sum(yest==y)/length(y);
% [tpp , fpp]                = basicroc(y , fx);
% auc_est                    = auroc(tpp', fpp');
%
% figure(1)
% plot(fpp , tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
% axis([-0.02 , 1.02 , -0.02 , 1.02])
%
%
% Example 2  With Haar features + Adaboosting
% ---------
%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'test' , 'negatives');
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.preview            = 0;
% options.standardize        = 1;
% options.Npos               = 3000;
% options.Nneg               = 5000;
% options.negmax_size        = 1000;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.00;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.probaswitchIneg    = 0.5;
% options.posscalemin        = 1;
% options.posscalemax        = 1;
% options.negscalemin        = 0.7;
% options.negscalemax        = 3;
% options.typefeat           = 0;
%
% load model_detector_haar_24x24.mat
% model.cascade              = [size(model.param , 2) ; 0];
%
% [fx , y]                   = eval_model_dataset(options , model);
% yest                       = sign(fx);
% accuracy                   = sum(yest==y)/length(y);
% [tpp , fpp]                = basicroc(y , fx);
% auc_est                    = auroc(tpp', fpp');
%
% figure(1)
% plot(fpp , tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
% axis([-0.02 , 1.02 , -0.02 , 1.02])
%
%
% Example 3  With MBLBP features + Adaboosting
% ---------
%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'test' , 'negatives');
% options.posext             = {'pgm'};
% options.negext             = {'jpg'};
% options.negmax_size        = 1000;
% options.preview            = 0;
% options.standardize        = 1;
% options.Npos               = 3000;
% options.Nneg               = 5000;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.00;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.probaswitchIneg    = 0.5;
% options.posscalemin        = 1;
% options.posscalemax        = 1;
% options.negscalemin        = 0.7;
% options.negscalemax        = 3;
% options.typefeat           = 1;
%
% load model_detector_mblbp_24x24_4.mat
% model.cascade              = [size(model.param , 2) ; 0];
%
% [fx , y]                   = eval_model_dataset(options , model);
% yest                       = sign(fx);
% accuracy                   = sum(yest==y)/length(y);
% [tpp , fpp]                = basicroc(y , fx);
% auc_est                    = auroc(tpp', fpp');
%
% figure(1)
% plot(fpp , tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
% axis([-0.02 , 1.02 , -0.02 , 1.02])



if(nargin < 1)
    options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
    options.posext             = {'png'};
    options.negext             = {'jpg'};
    options.negmax_size        = 1000; 
    options.Npostrain          = 10000;
    options.Nnegtrain          = 10000;
    options.Npostest           = 10000;
    options.Nnegtest           = 10000;
    options.seed               = 5489;
    options.resetseed          = 1;
    options.preview            = 0;
    options.standardize        = 1;
    options.probaflipIpos      = 0.5;
    options.probarotIpos       = 0.05;
    options.m_angle            = 0;
    options.sigma_angle        = 7^2;
    options.probaswitchIneg    = 0.9;
    options.posscalemin        = 0.25;
    options.posscalemax        = 1.75;
    options.negscalemin        = 0.7;
    options.negscalemax        = 3;
    options.typefeat           = 3;
end


if(~any(strcmp(fieldnames(options) , 'positives_path')))
    options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
end
if(~any(strcmp(fieldnames(options) , 'negatives_path')))
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
end
if(~any(strcmp(fieldnames(options) , 'posext')))
    options.posext             = {'png'};
end
if(~any(strcmp(fieldnames(options) , 'negext')))
    options.negext             = {'jpg'};
end
if(~any(strcmp(fieldnames(options) , 'negmax_size')))
    options.negmax_size        = 1000;
end
if(~any(strcmp(fieldnames(options) , 'Npos')))
    options.Npos               = 10000;
end
if(~any(strcmp(fieldnames(options) , 'Nneg')))
    options.Nneg               = 10000;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat           = 3;
end
if(~any(strcmp(fieldnames(options) , 'seed')))
    options.seed               = 5489;
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
if(~any(strcmp(fieldnames(options) , 'posscalemin')))
    options.posscalemin          = 0.25;
end
if(~any(strcmp(fieldnames(options) , 'posscalemax')))
    options.posscalemax          = 1.75;
end
if(~any(strcmp(fieldnames(options) , 'negscalemin')))
    options.posscalemin          = 0.7;
end
if(~any(strcmp(fieldnames(options) , 'negscalemax')))
    options.posscalemax          = 3;
end
if(~any(strcmp(fieldnames(options) , 'resetseed')))
    options.resetseed            = 1;
end


if(options.typefeat == 0)
    
    
    
end

if(options.typefeat == 3)
    if(model.cs_opt == 1)
        if(model.maptable == 0)
            model.Nbins             = 16;
        elseif(model.maptable == 1)
            model.Nbins             = 15;
        elseif(model.maptable == 2)
            model.Nbins             = 6;
        elseif(model.maptable == 3)
            model.Nbins             = 6;
        end
        model.improvedLBP           = 0;
    else
        if(model.maptable == 0)
            model.Nbins             = 256;
        elseif(model.maptable == 1)
            model.Nbins             = 59;
        elseif(model.maptable == 2)
            model.Nbins             = 36;
        elseif(model.maptable == 3)
            model.Nbins             = 10;
        end
    end
    
    model.nH               = sum(floor(((1 - model.spyr(:,1))./(model.spyr(:,3)) + 1)).*floor((1 - model.spyr(:,2))./(model.spyr(:,4)) + 1));
    model.nscale           = length(model.scale);
    options.d              = model.Nbins*(1+model.improvedLBP)*model.nH*model.nscale;
end


%% reset seed eventually %%

if(options.resetseed)
    RandStream.setDefaultStream(RandStream.create('mt19937ar','seed',options.seed));
end


Ntotal                      = options.Npos + options.Nneg;
fx                          = zeros(1 , Ntotal);
y                           = zeros(1 , Ntotal);

%% Generate Npos Positives examples from Xpos without replacement %%

options.std_angle      = sqrt(options.sigma_angle);

directory              = [];
for i = 1:length(options.posext)
    directory          = [directory ; dir(fullfile(options.positives_path , ['*.' options.posext{i}]))] ;
end
nbpos_photos           = length(directory);
index_posphotos        = randperm(nbpos_photos);

h                      = waitbar(0,sprintf('Generating positives' ));
set(h , 'name' , sprintf('Generating positives'));
co                     = 1;
cophotos               = 1;
if(options.preview)
    fig = figure(1);
    set(fig,'doublebuffer','on');
    colormap(gray)
end

while(co <= options.Npos)
    
    Ipos                       = imread(fullfile(options.positives_path , directory(index_posphotos(cophotos)).name));
    [Ny , Nx , Nz]             = size(Ipos);
    if(Nz == 3)
        Ipos                   = rgb2gray(Ipos);
    end
    
    %% Flip Positives %%    
    if(rand < options.probaflipIpos)
        Ipos                   = Ipos(: , end:-1:1) ;
    end
    %% Rotation positives
    if(rand < options.probarotIpos)
        Ipos                   = fast_rotate(Ipos , options.m_angle + options.std_angle*randn(1,1));
    end
    %% Scaling positives
    scale                      = options.posscalemin + (options.posscalemax-options.posscalemin)*rand;
    Nys                        = round(Ny*scale);
    Nxs                        = round(Nx*scale);
    Ipos                       = imresize(Ipos , [Nys , Nxs]);
     %% Standardize image %%
    if(options.standardize)
        dIpos                  = double(Ipos);
        stdIpos                = std(dIpos(:));
        if(stdIpos~=0)
            dIpos              = dIpos/stdIpos;
        end
        minIpos                = min(dIpos(:));
        maxIpos                = max(dIpos(:));
        Ipos                   = uint8(floor(255*(dIpos-minIpos)/(maxIpos-minIpos)));
    end
   
    
    if(options.typefeat == 0)
        fx(co)                 = eval_haar_subwindow(Ipos , model);
    end
    if(options.typefeat == 1)
        fx(co)                 = eval_mblbp_subwindow(Ipos , model);
    end
    if(options.typefeat == 3)
        fx(co)                 = eval_hmblbp_spyr_subwindow(Ipos , model);
    end
    y(co)                      = 1;
    if(options.preview)
        imagesc(Ipos);
    end
    waitbar(co/options.Npos , h , sprintf('#pos = %d' , co));
    drawnow
    co                         = co + 1;
    if(cophotos < nbpos_photos)
        cophotos               = cophotos + 1;
    else
        cophotos               = 1;
        index_posphotos        = randperm(nbpos_photos);
    end
    options.ny                 = Ny;
    options.nx                 = Nx;
end
close(h)


%% Generate Nneg Negatives examples from Negatives images with replacement %%

cophotos               = 1;
directory              = [];
for i = 1:length(options.negext)
    directory          = [directory ; dir(fullfile(options.negatives_path , ['*.' options.negext{i}]))] ;
end
nbneg_photos           = length(directory);
index_negphotos        = randperm(nbneg_photos);
Ineg                   = imread(fullfile(options.negatives_path , directory(index_negphotos(cophotos)).name));
[Ny , Nx , Nz]         = size(Ineg);
if(Nz == 3)
    Ineg               = rgb2gray(Ineg);
end

maxNyNx                = max([Ny Nx]);
if(maxNyNx > options.negmax_size)
    ratio              = options.negmax_size/maxNyNx;
    Ny                 = ceil(ratio*Ny);
    Nx                 = ceil(ratio*Nx);
    Ineg               = imresize(Ineg , [Ny Nx]);
end

if(min(Ny,Nx) < options.negscalemax*max(options.ny,options.nx))
    scalemax           = min(Ny,Nx)/max(options.ny,options.nx);
else
    scalemax           = options.negscalemax;
end


h                     = waitbar(0,sprintf('Generating negatives'));
set(h , 'name' , sprintf('Generating negatives'))
while(co <= Ntotal)
    if(rand < options.probaswitchIneg)
        if(cophotos < nbneg_photos)
            cophotos               = cophotos + 1;
        else
            cophotos               = 1;
            index_negphotos        = randperm(nbneg_photos);
        end
        
        Ineg                       = imread(fullfile(options.negatives_path , directory(index_negphotos(cophotos)).name));
        [Ny , Nx , Nz]             = size(Ineg);
        if(Nz == 3)
            Ineg                   = rgb2gray(Ineg);
        end
        if(min(Ny,Nx) < options.negscalemax*max(options.ny,options.nx))
            scalemax              = min(Ny,Nx)/max(options.ny,options.nx);
        end
    end
    
    %% Scaling Negatives 
  
    scale                          = options.negscalemin + (scalemax-options.negscalemin)*rand;
    nys                            = min(Ny , round(options.ny*scale));
    nxs                            = min(Nx , round(options.nx*scale));
    yy                             = ceil(1 + (Ny - nys - 1)*rand);
    xx                             = ceil(1 + (Nx - nxs - 1)*rand);
    Itemp                          = Ineg(yy:yy+nys-1 , xx:xx+nxs-1);
    Itemp                          = imresize(Itemp , [options.ny , options.nx]);
    
    %% standardize Negatives %%
    
    if(options.standardize)
        dIneg                      = double(Itemp);
        stdIneg                    = std(dIneg(:));
        if(stdIneg~=0)
            dIneg                  = dIneg/stdIneg;
        end
        minIneg                    = min(dIneg(:));
        maxIneg                    = max(dIneg(:));
        Itemp                      = uint8(floor(255*(dIneg-minIneg)/(maxIneg-minIneg)));
    end
   
    if(std(double(Itemp(:)),1)~= 0)
        if(options.typefeat == 0)
            fx(co)                 = eval_haar_subwindow(Itemp , model);
        end
        if(options.typefeat == 1)
            fx(co)                 = eval_mblbp_subwindow(Itemp , model);
        end
        if(options.typefeat == 3)
            fx(co)                 = eval_hmblbp_spyr_subwindow(Itemp , model);
        end
        y(co)                      = -1; 
        if(options.preview)
            imagesc(Ineg);
            hold on
            hh = rectangle('position' , [x , y , nxs , nys]);
            set(hh , 'linewidth' , 3 , 'edgecolor' , [1 0 0])
            hold off
        end
        waitbar((co-options.Npos)/options.Nneg , h , sprintf('#Neg = %d' , co-options.Npos));
        drawnow
        co                    = co + 1;
    end
end
close(h)
