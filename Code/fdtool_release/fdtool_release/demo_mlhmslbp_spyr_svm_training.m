%% Demo illustrating HMBLBP + HomkerMap + FastSVM train

clc,clear, close all, drawnow
load viola_24x24
Xviola                                 = X;
yviola                                 = y;
load jensen_24x24
X                                      = cat(3 , X , Xviola);
y                                      = [y , yviola];
[Ny , Nx , P]                          = size(X);
y                                      = int8(y);

%options.spyr                           = [1 , 1 , 1 , 1 , 1];
options.spyr                           = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
options.spyr                           = [1 , 1 , 1 , 1, 1];
options.scale                          = [1];
options.cs_opt                         = 0;
options.maptable                       = 0;
options.useFF                          = 0;
options.improvedLBP                    = 0;
options.rmextremebins                  = 1;
options.color                          = 0;
options.norm                           = [0,0,4];
options.clamp                          = 0.2;

options.n                              = 0;
options.L                              = 1.2;
options.kerneltype                     = 0;


options.nH                             = sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
options.nscale                         = length(options.scale);
options.ncolor                         = 1;

if(options.cs_opt == 1)
    if(options.maptable == 0)
        options.nbin                       = 16;
    elseif(options.maptable==1)
        options.nbin                       = 15;
    elseif(options.maptable==2)
        options.nbin                       = 6;
    elseif(options.maptable==3)
        options.nbin                       = 6;
    end
else
    if(options.maptable == 0)
        options.nbin                       = 256;
    elseif(options.maptable==1)
        options.nbin                       = 59;
    elseif(options.maptable==2)
        options.nbin                       = 36;
    elseif(options.maptable==3)
        options.nbin                       = 10;
    end
end

indp                                   = y==1;
indn                                   = ~indp;

X                                      = cat(3 , X(: , : , indp) , X(: , : , indn));
y                                      = [y(indp) , y(indn)];

N                                      = 4000;
vect                                   = [1:N , 9916+1:9916+1+N-1];
indextrain                             = vect(randperm(length(vect)));
indextest                              = (1:length(y));
indextest(indextrain)                  = [];

ytrain                                 = y(indextrain);
ytest                                  = y(indextest);

indp                                   = find(ytest == 1);
indn                                   = find(ytest ==-1);




%% mlhmslbp_spyr  Features%%

Htrain                                 = zeros(options.nbin*options.nH*options.nscale*options.ncolor , length(indextrain));
for i = 1 : length(indextrain)
    %   Htrain(: , i)                       = mlhmslbp_spyr(X(:,:,indextrain(i)) , options);
    [fxtemp , ytemp , Htrain(: , i)]    = eval_hmblbp_spyr_subwindow(X(:,:,indextrain(i)) , options);
end

if(options.n > 0)
    Htrain                             = homkermap(Htrain , options );
end

options.model                          = train_dense(double(ytrain)' , Htrain , '-s 2 -B 1 -c 100' , 'col');

Htest                                  = zeros(options.nbin*options.nH*options.nscale*options.ncolor , length(indextest));

for i = 1 : length(indextest)
    %   Htest(: , i)                       = mlhmslbp_spyr(X(:,:,indextest(i)) , options);
    [fxtemp , ytemp , Htest(: , i)]    = eval_hmblbp_spyr_subwindow(X(:,:,indextest(i)) , options);
end

if(options.n > 0)
    Htest                              = homkermap(Htest , options );
end

%[ytest_est , accuracy, fxtest]          = predict_dense(double(ytest'), Htest, options.model , '-b 0','col');

fxtest                                 = options.model.w(1:end-1)*Htest + options.model.w(end);
if(options.model.Label(1)==-1)
    fxtest                             = -fxtest;
end

ytest_est                               = sign(fxtest);
accuracy                                = sum(ytest_est == ytest)/length(ytest);

%ytest_est                               = ytest_est';
tp                                      = sum(ytest_est(indp) == ytest(indp))/length(indp);
fp                                      = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn);
perf                                    = sum(ytest_est == ytest)/length(ytest);

if(options.model.Label(1) == 1)
    [tpp , fpp]                         = basicroc(ytest , fxtest);
else
    [tpp , fpp]                         = basicroc(ytest , fxtest);
end

auc_est                                 = auroc(tpp', fpp');

figure(1)
plot(fpp , tpp  , 'b', 'linewidth' , 2)
grid on
title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
axis([-0.02 , 1.02 , -0.02 , 1.02])

%% Demo illustrating Trained HMBLBP for different size of the test images %


clear,close all
load model_hmblbp_R4



options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
options.posext             = {'png'};
options.negext             = {'jpg'};
options.preview            = 0;
options.Npostrain          = 0;
options.Nnegtrain          = 0;
options.Npostest           = 5000;
options.Nnegtest           = 5000;
options.probaflipIpos      = 0.5;
options.probarotIpos       = 0.05;
options.m_angle            = 0;
options.sigma_angle        = 5^2;
options.probaswitchIneg    = 0.9;
options.posscalemin        = 0.25;
options.posscalemax        = 1.75;
options.typefeat           = 3;
options.spyr               = [1 , 1 , 1 , 1  , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
options.scale              = [1];
options.maptable           = 0;
options.useFF              = 0;
options.cs_opt             = 1;
options.improvedLBP        = 0;
options.rmextremebins      = 0;
options.color              = 0;
options.norm               = [0,0,4];
options.clamp              = 0.2;


vect                       = 0.2:0.5:2.3;

tp                         = zeros(1 , length(vect));
fp                         = zeros(1 , length(vect));
perf                       = zeros(1 , length(vect));
auc_est                    = zeros(1 , length(vect));

co                         = 1;

for i = vect
    options.negscalemin                   = i;
    options.negscalemax                   = i;
    
    [Xtrain , ytrain , Xtest , ytest]     = generate_face_features(options);
    indp                                  = find(ytest == 1);
    indn                                  = find(ytest ==-1);
    fxtest                                = model.w(1:end-1)*Xtest + model.w(end);
    ytest_est                             = sign(fxtest);
    accuracy                              = sum(ytest_est == ytest)/length(ytest);
    tp(co)                                = sum(ytest_est(indp) == ytest(indp))/length(indp);
    fp(co)                                = 1 - sum(ytest_est(indn) == ytest(indn))/length(indn);
    perf(co)                              = sum(ytest_est == ytest)/length(ytest);
    [tpp , fpp]                           = basicroc(ytest , fxtest);
    auc_est(co)                           = auroc(tpp', fpp');
    co                                    = co + 1;
    
end

plot(round(vect*128) , perf);





%% Demo illustrating Trained HMBLBP for different size of the test images %


clear,close all



options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
options.posext             = {'png'};
options.negext             = {'jpg'};
options.preview            = 0;
options.Npostrain          = 0;
options.Nnegtrain          = 0;
options.Npostest           = 5000;
options.Nnegtest           = 5000;
options.probaflipIpos      = 0.5;
options.probarotIpos       = 0.05;
options.m_angle            = 0;
options.sigma_angle        = 5^2;
options.probaswitchIneg    = 0.9;
options.posscalemin        = 0.25;
options.posscalemax        = 1.75;
options.negscalemin        = 0.3;
options.negscalemax        = 1.5;
options.typefeat           = 2;
options.spyr               = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
options.scale              = [1];
options.maptable           = 0;
options.useFF              = 0;
options.cs_opt             = 1;
options.improvedLBP        = 0;
options.rmextremebins      = 0;
options.color              = 0;
options.norm               = [0,0,4];
options.clamp              = 0.2;


options.Npos                 = options.Npostrain + options.Npostest;
options.Nneg                 = options.Nnegtrain + options.Nnegtest;
options.Ntrain               = options.Npostrain + options.Nnegtrain;
options.Ntest                = options.Npostest  + options.Nnegtest;
options.std_angle            = sqrt(options.sigma_angle);

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
if(options.typefeat == 2)
    options.d               = options.Nbins*(1+options.improvedLBP)*options.nH*options.nscale;
end

Ntotal                      = options.Npos + options.Nneg;


model_hmblbp                = load ('model_hmblbp_R4.mat');
model_haar                  = load ('model_detector_haar_24x24.mat');

model_haar.model.cascade    = [size(model_haar.model.param , 2) ; 0];

%% Generate Npos Positives examples from Xpos without replacement %%


directory              = dir(fullfile(options.positives_path , ['*.' options.posext]));
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
    
    if(rand < options.probaflipIpos)
        Ipos                   = Ipos(: , end:-1:1) ;
    end
    if(rand < options.probarotIpos)
        Ipos                   = fast_rotate(Ipos , options.m_angle + options.std_angle*randn(1,1));
    end
    
    scale                      = options.posscalemin + (options.posscalemax-options.posscalemin)*rand;
    Nys                        = round(Ny*scale);
    Nxs                        = round(Nx*scale);
    Ipos                       = imresize(Ipos , [Nys , Nxs]);
 
    fx_hmblbp(co)              = eval_hmblbp_spyr_subwindow(Ipos , model_hmblbp.model);
    fx_haar(co)                = eval_haar_subwindow(Ipos , model_haar.model);

    
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

cophotos               = 1;
directory              = dir(fullfile(options.negatives_path , ['*.' options.negext]));
nbneg_photos           = length(directory);
index_negphotos        = randperm(nbneg_photos);
Ineg                   = imread(fullfile(options.negatives_path , directory(index_negphotos(cophotos)).name));
[Ny , Nx , Nz]         = size(Ineg);
if(Nz == 3)
    Ineg               = rgb2gray(Ineg);
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
    
    scale                          = options.negscalemin + (scalemax-options.negscalemin)*rand;
    nys                            = min(Ny , round(options.ny*scale));
    nxs                            = min(Nx , round(options.nx*scale));
    
    y                              = ceil(1 + (Ny - nys - 1)*rand);
    x                              = ceil(1 + (Nx - nxs - 1)*rand);
    Itemp                          = Ineg(y:y+nys-1 , x:x+nxs-1);
    Itemp                          = imresize(Itemp , [options.ny , options.nx]);
    
    if(std(double(Itemp(:)),1)~= 0)
        fx_hmblbp(co)             = eval_hmblbp_spyr_subwindow(Itemp , model_hmblbp.model);
        fx_haar(co)               = eval_haar_subwindow(Itemp , model_haar.model);
        
        waitbar((co-options.Npos)/options.Nneg , h , sprintf('#Neg = %d' , co-options.Npos));
        drawnow
        
        co                    = co + 1;
    end
end
close(h)

ytest_est_hmblbp                      = sign(fx_hmblbp);
ytest_est_haar                        = sign(fx_haar);

ytest                                 = ones(1 , options.Npostest+options.Nnegtest);
ytest(options.Npostest+1:options.Npostest+options.Nnegtest) = -1;

indp                                  = (1:options.Npostest);
indn                                  = (options.Npostest+1:options.Npostest+options.Nnegtest);


tp_hmblbp                             = sum(ytest_est_hmblbp(indp) == ytest(indp))/length(indp);
fp_hmblbp                             = 1 - sum(ytest_est_hmblbp(indn) == ytest(indn))/length(indn);
perf_hmblbp                           = sum(ytest_est_hmblbp == ytest)/length(ytest);
[tpp_hmblbp , fpp_hmblbp]             = basicroc(ytest , fx_hmblbp);
auc_est_hmblbp                        = auroc(tpp_hmblbp', fpp_hmblbp');
    


tp_haar                               = sum(ytest_est_haar(indp) == ytest(indp))/length(indp);
fp_haar                               = 1 - sum(ytest_est_haar(indn) == ytest(indn))/length(indn);
perf_haar                             = sum(ytest_est_haar == ytest)/length(ytest);
[tpp_haar , fpp_haar]                 = basicroc(ytest , fx_haar);
auc_est_haar                          = auroc(tpp_haar', fpp_haar');


figure(1)
plot(fpp_haar , tpp_haar , fpp_hmblbp , tpp_hmblbp , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
grid on
legend('Haar', 'HMBLBP' , 'location' , 'southeast')
title(sprintf('ROC Haar+boosting vs HMBLBP+SVM'))

figure(2)
plot(fx_haar)
title(sprintf('Haar+boosting'))


figure(3)
plot(fx_hmblbp)
title(sprintf('HMBLBP+SVM'))
