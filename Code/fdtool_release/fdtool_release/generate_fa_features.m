function  [Xfa , fxfa , pfa_current]     = generate_fa_features(options)
%
%
%  Generate new negative features considered as false alarms for the
%  current trained model
%
%  Usage
%  ------
%
%  [Xfa , fxfa , pfa_current]   = generate_fa_features(options)
%
%  Inputs
%  -------
%  options          Options struture
%
%                   negatives_path   Path from negative images are loaded for generating negative examples
%                   negext           Negatives extension files
%                   negmax_size      Maximum side of each negative image (default = 1000)
%                   seed             Seed value for random generator in order to generate the same positive sets
%                   resetseed        Reset generator with given seed (default resetseed = 1)
%                   preview          Preview current example (default preview = 0)
%                   standardize      Standardize images (default standardize = 1)
%                   Nnegboost        Number of desired negatives examples . Extracted by bilinear
%                   typefeat         Type of features (featype: 0 <=> Haar, 1 <=> MBLBP, 2 <=> Histogram of MBLBP, 3 <=> Histogram of MBLGP)
%  Outputs
%  -------
%
%  Xfa              features associated with False alarms for current model (d x Nnegboost)
%  fxfa             Output predicted values of the false alarms (1 x Nnegboost)
%  pfa_current      Current false alarms probability
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 02/25/2011
%
%

if(nargin < 1)
    options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
    options.negext             = {'jpg'};
    options.negmax_size        = 1000;
    options.Nnegboost          = 1000;
    options.seed               = 5489;
    options.resetseed          = 1;
    options.preview            = 0;
    
    options.typefeat           = 2;
    
    options.addbias            = 1;
    options.max_detections     = 5000;
    options.num_threads        = -1;
    options.dimsIscan          = [24 , 24];
    options.scalingbox         = [2 , 1.4 , 1.8];
    options.mergingbox         = [1/2 , 1/2 , 0.8];
    options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
    options.scale              = [1];
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
    
    options.dimsItraining      = [24 , 24];
    options.rect_param         = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
    options.usesingle          = 1;
    options.transpose          = 0;
    
    
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
if(~any(strcmp(fieldnames(options) , 'Nnegboost')))
    options.Nnegboost          = 1000;
end
if(~any(strcmp(fieldnames(options) , 'seed')))
    options.seed                = 5489;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat            = 3;
end
if(~any(strcmp(fieldnames(options) , 'addbias')))
    options.addbias            = 1;
end
if(~any(strcmp(fieldnames(options) , 'boost_ite')))
    options.boost_ite          = 10;
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
if(~any(strcmp(fieldnames(options) , 'max_detections')))
    options.max_detections     = 5000;
end
if(~any(strcmp(fieldnames(options) , 'num_threads')))
    options.num_threads        = -1;
end
if(~any(strcmp(fieldnames(options) , 'dimsIscan')))
    options.dimsIscan          = [24 , 24];
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
if(~any(strcmp(fieldnames(options) , 'resetseed')))
    options.resetseed            = 1;
end
if(~any(strcmp(fieldnames(options) , 'typefeat')))
    options.typefeat             = 2;
end


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
    options.nH                  = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
    options.nscale              = length(options.scale);
    options.d                   = options.Nbins*(1+options.improvedLBP)*options.nH*options.nscale;
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

%% reset seed eventually %%

if(options.resetseed)
    RandStream.setDefaultStream(RandStream.create('mt19937ar','seed',options.seed));
end

Ntotal                   = options.Nnegboost;

if(options.typefeat == 0)
    if(options.transpose)
        if(options.usesingle)
            Xfa            = zeros(Ntotal , options.d , 'single');
        else
            Xfa            = zeros(Ntotal , options.d);
        end
    else
        if(options.usesingle)
            Xfa            = zeros(options.d , Ntotal , 'single');
        else
            Xfa            = zeros(options.d , Ntotal);
        end
    end
elseif(options.typefeat == 1)
    if(options.transpose)
        if(options.usesingle)
            Xfa            = zeros(Ntotal , options.d , 'single');
        else
            Xfa            = zeros(Ntotal , options.d);
        end
    else
        if(options.usesingle)
            Xfa            = zeros(options.d , Ntotal , 'single');
        else
            Xfa            = zeros(options.d , Ntotal);
        end
    end
elseif((options.typefeat == 2) || (options.typefeat == 3))
    Xfa                 = zeros(options.d , Ntotal);
end

fxfa                     = zeros(1 , Ntotal);
options.postprocessing   = 0;

%% Generate Nneg Negatives examples from Negatives images with replacement %%

cophotos               = 1;
co                     = 1;
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
if(options.preview)
    imagesc(Ineg);
end

cte                    = 1;
cumstat                = zeros(1 , 2);
h                      = waitbar(0,sprintf('Generating negatives'));
set(h , 'name' , sprintf('Boosting negatives  stage %d/%d' , options.m , options.boost_ite))
drawnow

if(options.preview)
    fig = figure(1);
    set(fig,'doublebuffer','on');
    colormap(gray)
end
while(co <= Ntotal)
    if(options.typefeat == 0)
        [D , stat]                              = detector_haar(Ineg , options);
        cumstat                                 = cumstat + stat;
        for i = 1:min(Ntotal,size(D,2))
            Icrop                               = imresize(Ineg(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1) , options.dimsItraining);
            if(options.transpose)
                Xfa(co , :)                     = haar(Icrop , options);
            else
                Xfa(: , co)                     = haar(Icrop , options);
            end
            %            [fxfa(co) , yfxfa ]                 = eval_haar(Icrop , options);
            fxfa(co)                            = D(5 , i);
            co                                  = co + 1;
        end
    elseif(options.typefeat == 1)
        [D , stat]                              = detector_mblbp(Ineg , options);
        cumstat                                 = cumstat + stat;
        for i = 1:min(Ntotal,size(D,2))
            Icrop                               = imresize(Ineg(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1) , options.dimsItraining);
            if(options.transpose)
                Xfa(co,:)                       = mblbp(Icrop , options);
            else
                Xfa(: , co)                     = mblbp(Icrop , options);
            end
            %           [fxfa(co) , yfxfa ]                 = eval_mblbp(Icrop , options);
            fxfa(co)                           = D(5 , i);
            co                                  = co + 1;
        end
    elseif(options.typefeat == 2)
        [D , stat]                              = detector_mlhmslbp_spyr(Ineg , options);
        cumstat                                 = cumstat + stat;
        for i = 1:min(Ntotal,size(D,2))
            Icrop                               = Ineg(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1);
            [fxfa(co) , yfxfa  , Xfa(: , co) ]  = eval_hmblbp_spyr_subwindow(Icrop , options);
            co                                  = co + 1;
        end
    elseif(options.typefeat == 3)
        [D , stat]                              = detector_mlhmslgp_spyr(Ineg , options);
        cumstat                                 = cumstat + stat;
        for i = 1:min(Ntotal,size(D,2))
            Icrop                               = Ineg(D(2,i)-(1-cte)*1:D(2,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1 , D(1,i)-(1-cte)*1:D(1,i)-(1-cte)*1+D(3,i)+(1-cte)*2-1);
            [fxfa(co) , yfxfa  , Xfa(: , co) ]  = eval_hmblgp_spyr_subwindow(Icrop , options);
            co                                  = co + 1;
        end
    end
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
    if(options.preview)
        imagesc(Ineg);
        plot_rectangle(D)
    end
    waitbar((co - 1)/Ntotal , h , sprintf('#Neg = %d, Pfa = %10.9f' , co , cumstat(1)/cumstat(2)));
    drawnow
end
close(h)

pfa_current                     = cumstat(1)/cumstat(2);
if((options.transpose) && (options.typefeat < 2))
    Xfa                         = Xfa(1:Ntotal,:);
else
    Xfa                         = Xfa(:,1:Ntotal);
end
fxfa                            = fxfa(1:Ntotal);