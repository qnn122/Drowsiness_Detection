function  [Xtrain , ytrain , Xtest , ytest] = generate_face_features(options)

%
%  Generate i.i.d positive and negative features from positves and
%  negatives folder respectively
%
%  Usage
%  ------
%
%  [Xtrain , ytrain , Xtest , ytest]         = generate_face_features(options)
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
%                   seed             Seed value for random generator in order to generate the same positive sets
%                   resetseed        Reset generator with given seed (default resetseed = 1)
%                   preview          Preview current example (default preview = 0)
%                   standardize      Standardize images (default standardize = 1)
%                   Npostrain        Number of desired positives examples for training set (Npostrain+Npostest <= Npostotal)
%                   Nnegtrain        Number of desired negatives examples for training set. Extracted by bilinear
%                                    interpolation from the negatives database
%                   Npostest         Number of desired positives examples for testing set (Npostrain+Npostest <= Npostotal)
%                   Nnegtest         Number of desired negatives examples for testing set. Extracted by bilinear
%                   probaflipIpos    Probability to flip Positive examples (default probaflipIpos = 0.5)
%                   probarotIpos     Probability to rotate Positives examples with an angle~N(m_angle,sigma_angle) (default probarotIpos = 0.01)
%                   m_angle          Mean rotation angle value in degree (default mangle = 0)
%                   sigma_angle      variance of the rotation angle value (default sigma_angle = 5^2)
%                   probaswitchIneg  Probability to swith from another picture in the negatives database (default probaswitchIneg = 0.005)
%                   posscalemin      Minimum scaling factor to apply on positives patch subwindows (default scalemin = 0.25)
%                   posscalemax      Maximum scaling factor to apply on positives patch subwindows (default scalemax = 2)
%                   negscalemin      Minimum scaling factor to apply on negatives patch subwindows (default scalemin = 1)
%                   negscalemax      Maximum scaling factor to apply on negatives patch subwindows (default scalemax = 5)
%                   typefeat         Type of features (featype: 0 <=> Haar, 1 <=> MBLBP, 2 <=> Histogram of MBLBP, 3 <=> Histogram of MBLGP)
%
%  Outputs
%  -------
%
%  Xtrain           Train features generated from picts (d x (options.Npostrain+options.Nnegtrain))
%  ytrain           Train label vector (1 x (options.Npostrain+options.Nnegtrain)) with yi=+1 for faces, yi=-1 else
%  Xtest            Test  features generated from picts (d x (options.Npostest+options.Nnegtest))
%  ytest            Test  label vector (1 x (options.Npostest+options.Nnegtest)) with yi=+1 for faces, yi=-1 else
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 02/25/2011
%
%
% close all
% options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
% options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
% options.posext             = 'png';
% options.negext             = 'jpg';
% options.negmax_size        = 1000;
% options.preview            = 0;
% options.Npostrain          = 15000;
% options.Nnegtrain          = 30000;
% options.Npostest           = 5000;
% options.Nnegtest           = 5000;
% options.probaflipIpos      = 0.5;
% options.probarotIpos       = 0.05;
% options.m_angle            = 0;
% options.sigma_angle        = 5^2;
% options.seed               = 5489;
% options.resetseed          = 1;
% options.probaswitchIneg    = 0.9;
% options.posscalemin        = 0.25;
% options.posscalemax        = 1.75;
% options.negscalemin        = 0.7;
% options.negscalemax        = 3;
% options.typefeat           = 3;
% options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
% options.scale              = [1];
% options.maptable           = 0;
% options.useFF              = 0;
% options.cs_opt             = 1;
% options.improvedLBP        = 0;
% options.rmextremebins      = 1;
% options.color              = 0;
% options.norm               = 4;
% options.clamp              = 0.2;
% options.s                  = 2;
% options.B                  = 1;
% options.c                  = 5;
%
% [Xtrain , ytrain , Xtest , ytest]     = generate_face_features(options);
% indp                                  = find(ytest == 1);
% indn                                  = find(ytest ==-1);
% wpos                                  = options.Nnegtrain/options.Npostrain;
% options.model                         = train_dense(ytrain' , Xtrain , sprintf('-s %d -B %d -c %d' , options.s , options.B , options.c) , 'col');
%% options.model                         = train_dense(ytrain' , Xtrain , sprintf('-s %d -B %d -c %d -w1 %f' , options.s , options.B , options.c , wpos) , 'col');
% fxtest                                = options.model.w(1:end-1)*Xtest + options.model.w(end);
% if(options.model.Label(1)==-1)
%     fxtest                            = -fxtest;
% end
% ytest_est                             = sign(fxtest);
% accuracy                              = sum(ytest_est == ytest)/length(ytest);
% tp                                    = sum(ytest_est(indp) == ytest(indp))/length(indp);
% fp                                    = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn);
% perf                                  = sum(ytest_est == ytest)/length(ytest);
%
% if(options.model.Label(1) == 1)
%     [tpp , fpp]                       = basicroc(ytest , fxtest);
% else
%     [tpp , fpp]                       = basicroc(ytest , fxtest);
% end
%
% auc_est                               = auroc(tpp', fpp');
%
% figure(1)
% plot(fpp , tpp  , 'b', 'linewidth' , 2)
% grid on
% title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
% axis([-0.02 , 1.02 , -0.02 , 1.02])
%
% figure(2)
% plot(options.model.w)
%
% w = options.model.w;
% save modelw8 w
%


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
    
    options.typefeat           = 2;
    
    options.addbias            = 1;
    options.num_threads        = -1;
    options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
    options.scale              = 1;
    options.maptable           = 0;
    options.useFF              = 0;
    options.cs_opt             = 1;
    options.improvedLBP        = 0;
    options.improvedLGP        = 0;
    options.rmextremebins      = 0;
    options.color              = 0;
    options.norm               = 2;
    options.clamp              = 0.2;
    options.n                  = 0;
    options.L                  = 1.2;
    options.kerneltype         = 0;
    options.numsubdiv          = 8;
    options.minexponent        = -20;
    options.maxexponent        = 8;
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
    options.negext             = {'jpg'};
end
if(~any(strcmp(fieldnames(options) , 'seed')))
    options.seed               = 5489;
end
if(~any(strcmp(fieldnames(options) , 'resetseed')))
    options.resetseed          = 1;
end
if(~any(strcmp(fieldnames(options) , 'standardize')))
    options.standardize        = 1;
end
if(~any(strcmp(fieldnames(options) , 'preview')))
    options.preview            = 0;
end
if(~any(strcmp(fieldnames(options) , 'negmax_size')))
    options.negmax_size        = 1000;
end
if(~any(strcmp(fieldnames(options) , 'Npostrain')))
    options.Npostrain          = 5000;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtrain')))
    options.Nnegtrain          = 10000;
end
if(~any(strcmp(fieldnames(options) , 'Npostest')))
    options.Npostest           = 5000;
end
if(~any(strcmp(fieldnames(options) , 'Nnegtest')))
    options.Nnegtest           = 5000;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat           = 3;
end
if(~any(strcmp(fieldnames(options) , 'addbias')))
    options.addbias            = 1;
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
if(~any(strcmp(fieldnames(options) , 'n')))
    options.n                  = 0;
end
if(~any(strcmp(fieldnames(options) , 'L')))
    options.L                  = 0.7;
end
if(~any(strcmp(fieldnames(options) , 'kerneltype')))
    options.kerneltype         = 0;
end
if(~any(strcmp(fieldnames(options) , 'numsubdiv')))
    options.numsubdiv          = 8;
end
if(~any(strcmp(fieldnames(options) , 'minexponent')))
    options.numsubdiv          = -20;
end
if(~any(strcmp(fieldnames(options) , 'maxexponent')))
    options.maxexponent        = 8;
end
if(~any(strcmp(fieldnames(options) , 'num_threads')))
    options.num_threads        = -1;
end
if(~any(strcmp(fieldnames(options) , 'spyr')))
    options.spyr                = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
end
if(~any(strcmp(fieldnames(options) , 'scale')))
    options.scale                = 1;
end
if(~any(strcmp(fieldnames(options) , 'maptable')))
    options.maptable             = 0;
end
if(~any(strcmp(fieldnames(options) , 'useFF')))
    options.useFF                = 0;
end
if(~any(strcmp(fieldnames(options) , 'cs_opt')))
    options.cs_opt               = 1;
end
if(~any(strcmp(fieldnames(options) , 'improvedLBP')))
    options.improvedLBP          = 0;
end
if(~any(strcmp(fieldnames(options) , 'improvedLGP')))
    options.improvedLGP          = 0;
end
if(~any(strcmp(fieldnames(options) , 'rmextremebins')))
    options.rmextremebins        = 0;
end
if(~any(strcmp(fieldnames(options) , 'color')))
    options.color                = 0;
end
if(~any(strcmp(fieldnames(options) , 'norm')))
    options.norm                 = 2;
end
if(~any(strcmp(fieldnames(options) , 'clamp')))
    options.clamp                = 0.2;
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
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat             = 2;
end



options.Npos                 = options.Npostrain + options.Npostest;
options.Nneg                 = options.Nnegtrain + options.Nnegtest;
options.Ntrain               = options.Npostrain + options.Nnegtrain;
options.Ntest                = options.Npostest  + options.Nnegtest;
options.std_angle            = sqrt(options.sigma_angle);

if(options.typefeat == 0)
    options.d                       = size(options.F , 2);
elseif(options.typefeat == 1)
    options.d                       = size(options.F , 2);
elseif(options.typefeat == 2)
    if(options.cs_opt == 1)
        if(options.maptable == 0)
            options.Nbins           = 16;
        elseif(options.maptable == 1)
            options.Nbins           = 15;
        elseif(options.maptable == 2)
            options.Nbins           = 6;
        elseif(options.maptable == 3)
            options.Nbins           = 6;
        end
        options.improvedLBP         = 0;
    else
        if(options.maptable == 0)
            options.Nbins           = 256;
        elseif(options.maptable == 1)
            options.Nbins           = 59;
        elseif(options.maptable == 2)
            options.Nbins           = 36;
        elseif(options.maptable == 3)
            options.Nbins           = 10;
        end
    end
    options.nH              = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
    options.nscale          = length(options.scale);
    options.d               = options.Nbins*(1+options.improvedLBP)*options.nH*options.nscale;
elseif((options.typefeat == 3))
    if(options.cs_opt == 1)
        if(options.maptable == 0)
            options.Nbins           = 16;
        elseif(options.maptable == 1)
            options.Nbins           = 15;
        elseif(options.maptable == 2)
            options.Nbins           = 6;
        elseif(options.maptable == 3)
            options.Nbins           = 6;
        end
        options.improvedLGP         = 0;
    else
        if(options.maptable == 0)
            options.Nbins           = 256;
        elseif(options.maptable == 1)
            options.Nbins           = 59;
        elseif(options.maptable == 2)
            options.Nbins           = 36;
        elseif(options.maptable == 3)
            options.Nbins           = 10;
        end
    end
    options.nH              = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
    options.nscale          = length(options.scale);
    options.d               = options.Nbins*(1+options.improvedLGP)*options.nH*options.nscale;
end

%%
Ntotal                      = options.Npos + options.Nneg;

if(options.typefeat == 0)
    if(options.transpose)
        if(options.usesingle)
            X               = zeros(Ntotal , options.d  , 'single');
        else
            X               = zeros(Ntotal , options.d);
        end
    else
        if(options.usesingle)
            X               = zeros(options.d , Ntotal , 'single');
        else
            X               = zeros(options.d , Ntotal);
        end
    end
elseif(options.typefeat == 1)
    if(options.transpose)
        if(options.usesingle)
            X               = zeros(Ntotal , options.d  , 'single');
        else
            X               = zeros(Ntotal , options.d);
        end
    else
        if(options.usesingle)
            X                   = zeros(options.d , Ntotal , 'single');
        else
            X                   = zeros(options.d , Ntotal);
        end
    end
elseif((options.typefeat == 2) || (options.typefeat == 3))
    X                       = zeros(options.d , Ntotal);
end

%% Init random generator %%

if(options.resetseed)
    RandStream.setDefaultStream(RandStream.create('mt19937ar','seed',options.seed));
end


%% Generate Npos Positives examples from Xpos without replacement %%

directory              = [];
for i = 1:length(options.posext)
    directory          = [directory ; dir(fullfile(options.positives_path , ['*.' options.posext{i}]))] ;
end
nbpos_photos           = length(directory);

if(nbpos_photos < 1)
    error('Positives directory is empty')
end

index_posphotos        = randperm(nbpos_photos);

h                      = waitbar(0,sprintf('Generating positives' ));
set(h , 'name' , sprintf('Generating positives'));
drawnow
co                     = 1;
cophotos               = 1;
if(options.preview)
    fig = figure(1);
    set(fig,'doublebuffer','on');
    colormap(gray)
end

while(co <= options.Npos)
    
    Ipos                         = imread(fullfile(options.positives_path , directory(index_posphotos(cophotos)).name));
    [Ny , Nx , Nz]               = size(Ipos);
    if(Nz == 3)
        Ipos                     = rgb2gray(Ipos);
    end
    
    %% Flip positives %%
    if(rand < options.probaflipIpos)
        Ipos                     = Ipos(: , end:-1:1) ;
    end
    
    %% Rotation positives
    if(rand < options.probarotIpos)
        Ipos                     = fast_rotate(Ipos , options.m_angle + options.std_angle*randn(1,1));
    end
    
    %% scaling positive %%
    scale                        = options.posscalemin + (options.posscalemax-options.posscalemin)*rand;
    Nys                          = round(Ny*scale);
    Nxs                          = round(Nx*scale);
    Ipos                         = imresize(Ipos , [Nys , Nxs]);
    
    if((options.typefeat == 0) || (options.typefeat == 1))
        Ipos                 = imresize(Ipos , options.dimsItraining);
    end
    
    %% Standardize image %%
    if(options.standardize)
        dIpos                      = double(Ipos);
        stdIpos                    = std(dIpos(:));
        if(stdIpos~=0)
            dIpos                  = dIpos/stdIpos;
        end
        minIpos                    = min(dIpos(:));
        maxIpos                    = max(dIpos(:));
        Ipos                       = uint8(floor(255*(dIpos-minIpos)/(maxIpos-minIpos)));
    end
    
    if(options.typefeat == 0)
        if(options.transpose)
            X(co , :)              = haar(Ipos , options);
        else
            X(: , co)              = haar(Ipos , options);
        end
    elseif(options.typefeat == 1)
        if(options.transpose)
            X(co , :)              = mblbp(Ipos , options);
        else
            X(: , co)              = mblbp(Ipos , options);
        end
    elseif(options.typefeat == 2)
        [fx , ytemp , X(: , co)]   = eval_hmblbp_spyr_subwindow(Ipos , options);
    elseif(options.typefeat == 3)
        [fx , ytemp , X(: , co)]   = eval_hmblgp_spyr_subwindow(Ipos , options);
    end
    if(options.preview)
        imagesc(Ipos);
    end
    waitbar(co/options.Npos , h , sprintf('#pos = %d' , co));
    drawnow
    co                           = co + 1;
    if(cophotos < nbpos_photos)
        cophotos                 = cophotos + 1;
    else
        cophotos                 = 1;
        index_posphotos          = randperm(nbpos_photos);
    end
    options.ny                   = Ny;
    options.nx                   = Nx;
    
end
close(h)

%% Generate Nneg Negatives examples from Negatives images with replacement %%

cophotos               = 1;
directory              = [];
for i = 1:length(options.negext)
    directory          = [directory ; dir(fullfile(options.negatives_path , ['*.' options.negext{i}]))] ;
end
nbneg_photos           = length(directory);
if(nbneg_photos < 1)
    error('Negatives directory is empty')
end

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
            scalemax               = min(Ny,Nx)/max(options.ny,options.nx);
        end
    end
    
    %% Scaling Negatives %%
    
    scale                          = options.negscalemin + (scalemax-options.negscalemin)*rand;
    nys                            = min(Ny , round(options.ny*scale));
    nxs                            = min(Nx , round(options.nx*scale));
    y                              = ceil(1 + (Ny - nys - 1)*rand);
    x                              = ceil(1 + (Nx - nxs - 1)*rand);
    Itemp                          = Ineg(y:y+nys-1 , x:x+nxs-1);
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
    
    if((options.typefeat == 0) || (options.typefeat == 1))
        Itemp                      = imresize(Itemp , options.dimsItraining);
    end
    
    if(std(double(Itemp(:)),1)~= 0)
        if(options.typefeat == 0)
            if(options.transpose)
                X(co ,:)             = haar(Itemp , options);
            else
                X(: , co)            = haar(Itemp , options);
            end
        elseif(options.typefeat == 1)
            if(options.transpose)
                X(co,:)              = mblbp(Itemp , options);
            else
                X(: , co)            = mblbp(Itemp , options);
            end
        elseif(options.typefeat == 2)
            [fx , ytemp , X(: , co)] = eval_hmblbp_spyr_subwindow(Itemp , options);
        elseif(options.typefeat == 3)
            [fx , ytemp , X(: , co)] = eval_hmblgp_spyr_subwindow(Itemp , options);
        end
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
if((options.transpose) && (options.typefeat < 2))
    Xtrain                                 = X([1:options.Npostrain,options.Npos+1:options.Npos+options.Nnegtrain],:);
else
    Xtrain                                 = X(:,[1:options.Npostrain,options.Npos+1:options.Npos+options.Nnegtrain]);
end
ytrain                                     = ones(1 , options.Ntrain);
ytrain(options.Npostrain+1:options.Ntrain) = -1;
if((options.transpose) && (options.typefeat < 2))
    Xtest                                  = X([options.Npostrain+1:options.Npos,options.Npos+options.Nnegtrain+1:Ntotal],:);
else
    Xtest                                  = X(:,[options.Npostrain+1:options.Npos,options.Npos+options.Nnegtrain+1:Ntotal]);
end
ytest                                      = ones(1 , options.Ntest);
ytest(options.Npostest+1:options.Ntest)    = -1;
