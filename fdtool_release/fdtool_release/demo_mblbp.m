%% Demo illustrating (Center-Symetric) MultiBlock Local Binary Pattern (mblbp)
%   (Center-Symetric) MultiBlock Local Binary Pattern
%
%   Usage
%   ------
%
%   z         =     mblbp(I , [options]);
%
%
%   Inputs
%   -------
%
%   I                                    Images (Ny x Nx x N) in UINT8 format (unsigned char)
%
%   options
%
%           F                            Features lists (5 x nF) int UINT32 where nF design the total number of mblbp features (see mblbp_featlist function)
%                                        F(: , i) = [if ; xf ; yf ; wf ; hf] where
% 									     if       index of the current feature, if = [1,...,nF]
% 									     xf,yf    coordinates of the current feature (top-left rectangle)
%                                        wf,hf    width and height of each of the 9 rectangles
%
%           map                          Feature's mapping vector in UINT8 format (unsigned char) (default map = 0:255)
%
%
%           cs_opt                       Center-Symetric LBP : 1 for computing CS-MBLBP features, 0 : for MBLBP (default cs_opt = 0)
%
%           a                            Tolerance (default a = 0)
%
% 		    transpose                    Transpose Output if tranpose = 1 (in order to speed up Boosting algorithm, default tranpose = 0)
%
%
% If compiled with the "OMP" compilation flag
%
% 	        num_threads                  Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)
%
%
%   Outputs
%   -------
%
%   z                                    MultiBlock LPB vector (nF x N) in UINT8 format for each positions (y,x) in [1+h,...,ny-h]x[1+w,...,nx-w] and (w,h) integral block size.

%% First example : compute MBLBP and CS-MBLBP Features on image I for a scale = 5

clear, close all
I                 = imread('0000_-12_0_0_15_0_1.pgm');
[Ny , Nx]         = size(I);
N                 = 8;
scale             = 5*[1  ; 1 ];

options.F         = mblbp_featlist(Ny , Nx , scale);
options.map       = uint8(0:2^N-1);
z1                = mblbp(I , options);
options.map       = uint8(0:2^(N/2)-1);
options.cs_opt    = 1;

z2                = mblbp(I , options);
template          = options.F(: , 1);

Nxx               = (Nx-3*template(4) + 1);
Nyy               = (Ny-3*template(5) + 1);

Imblbp1           = reshape(z1 , Nyy , Nxx);
Imblbp2           = reshape(z2 , Nyy , Nxx);

figure

subplot(131)
imagesc(I);
title('Original Image');
axis square


subplot(132)
imagesc(imresize(Imblbp1 , [Ny , Nx]))
title(sprintf('MBLBP with s = %d' , scale(1)));
axis square

subplot(133)
imagesc(imresize(Imblbp2 , [Ny , Nx]))
title(sprintf('CSMBLBP with s = %d' , scale(1)));
axis square


colormap(gray)

disp('Press key to continue')
pause

%% Second example : compute MBLBP and CS-MBLBP Features on image I for scale = {1,2}


Ny                  = 24;
Nx                  = 24;
N                   = 8;
Nimage              = 200;

scale               = [1 , 2 ; 1 , 2];
options.cs_opt      = 0;

load viola_24x24

I                   = X(: , : , Nimage);

options.F           = mblbp_featlist(Ny , Nx , scale);
% mapping1  = getmapping(N,'u2');
% map1      = uint8(mapping1.table);
options.map         = uint8(0:2^N-1);
z1                  = mblbp(I , options);

% mapping2  = getmapping(N/2,'u2');
% map2      = uint8(mapping2.table);
options.map         = uint8(0:2^(N/2)-1);
options.cs_opt      = 1;

z2                  = mblbp(I , options );


ind1                = find(options.F(1 , :) == 1);
template1           = options.F(: , ind1(1));
Nxx1                = (Nx-3*template1(4) + 1);
Nyy1                = (Ny-3*template1(5) + 1);
Imblbp10            = reshape(z1(ind1) , Nyy1 , Nxx1);
Imblbp11            = reshape(z2(ind1) , Nyy1 , Nxx1);


ind2                = find(options.F(1 , :) == 2);
template2           = options.F(: , ind2(1));
Nxx2                = (Nx-3*template2(4) + 1);
Nyy2                = (Ny-3*template2(5) + 1);
Imblbp20            = reshape(z1(ind2) , Nyy2 , Nxx2);
Imblbp21            = reshape(z2(ind2) , Nyy2 , Nxx2);


figure
subplot(231)
imagesc(I);
title('Original Image');
colorbar
axis square

subplot(232)
imagesc(imresize(Imblbp10 , [Ny , Nx]))
title(sprintf('MBLBP with s = %d' , scale(1,1)));
colorbar
axis square

subplot(233)
imagesc(imresize(Imblbp20 , [Ny , Nx]))
title(sprintf('MBLBP with s = %d' , scale(1,2)));
colorbar
axis square


subplot(234)
imagesc(I);
title('Original Image');
colorbar
axis square

subplot(235)
imagesc(imresize(Imblbp11 , [Ny , Nx]))
title(sprintf('CSMBLBP with s = %d' , scale(1,1)));
colorbar
axis square

subplot(236)
imagesc(imresize(Imblbp21 , [Ny , Nx]))
title(sprintf('CSMBLBP with s = %d' , scale(1,2)));
colorbar
axis square

colormap(gray)

disp('Press key to continue')
pause

%% Third example : Display best MBLBP Features from Gentleboosting & Adaboosting

load viola_24x24

y                     = int8(y);
indp                  = find(y == 1);
indn                  = find(y ==-1);


Ny                    = 24;
Nx                    = 24;
N                     = 8;
Nimage                = 110;
mapping               = getmapping(N,'u2');

options.T             = 3;
options.F             = mblbp_featlist(Ny , Nx);
%options.map           = uint8(mapping.table);
options.map           = uint8(0:255);

H                     = mblbp(X , options);


figure
imagesc(H)
title('MBLBP Features')
drawnow

index                 = randperm(length(y)); %shuffle data to avoid numerical discrepancies with long sequence of same label


tic,options.param     = mblbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options);,toc
[yest1 , fx1]         = mblbp_gentleboost_binary_predict_cascade(H , options);


figure
best_feats            = double(options.F(: , options.param(1 , 1:options.T)));
I                     = X(: , : , Nimage);
imagesc(I)
hold on
for i = 1:options.T
    h = rectangle('Position', [best_feats(2,i)-best_feats(4,i) + 0.5,  best_feats(3,i)-best_feats(5,i) + 0.5 ,  3*best_feats(4,i) ,  3*best_feats(5,i)]);
    set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
end
hold off
title(sprintf('Best %d MBLBP features from Gentleboosting' , options.T) , 'fontsize' , 13)
colormap(gray)


tic,options.param     = mblbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);,toc
[yest2 , fx2]         = mblbp_adaboost_binary_predict_cascade(H , options);


figure
best_feats            = double(options.F(: , options.param(1 , 1:options.T)));
I                     = X(: , : , Nimage);
imagesc(I)
hold on
for i = 1:options.T
    h = rectangle('Position', [best_feats(2,i)-best_feats(4,i) + 0.5,  best_feats(3,i)-best_feats(5,i) + 0.5 ,  3*best_feats(4,i) ,  3*best_feats(5,i)]);
    set(h , 'linewidth' , 2 , 'EdgeColor' , [1 0 0])
end
hold off
title(sprintf('Best %d MBLBP features from Adaboosting' , options.T) , 'fontsize' , 13)
colormap(gray)

disp('Press key to continue')
pause


%% Fourth example : compute MBLBP Features + Adaboosting with T weaklearners (Decision Stump)


load viola_24x24

y                          = int8(y);
indp                       = find(y == 1);
indn                       = find(y ==-1);


Ny                         = 24;
Nx                         = 24;
options.F                  = mblbp_featlist(Ny , Nx);
% mapping                    = getmapping(N,'u2');
% options.map              = uint8(mapping.table);
options.map                = uint8(0:255);
options.T                  = 50;


H                          = mblbp(X , options);
figure
imagesc(H)
title('MBLBP Features')
drawnow



index                      = randperm(length(y));  %shuffle data to avoid numerical discrepancies with long sequence of same label


options.param              = mblbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);
[yest_train , fx_train]    = mblbp_adaboost_binary_predict_cascade(H , options);

tp_train                   = sum(yest_train(indp) == y(indp))/length(indp)
fp_train                   = 1 - sum(yest_train(indn) == y(indn))/length(indn)
Perf_train                 = sum(yest_train == y)/length(y)
[tpp_train , fpp_train]    = basicroc(y , fx_train);

[dum , ind]                = sort(y , 'descend');
figure
plot(fx_train(ind))
title(sprintf('Output of the strong classifier for train data with T = %d' , options.T))


load jensen_24x24

y                          = int8(y);
indp                       = find(y == 1);
indn                       = find(y ==-1);

H                          = mblbp(X , options);
[yest_test , fx_test]      = mblbp_adaboost_binary_predict_cascade(H , options);

tp_test                    = sum(yest_test(indp) == y(indp))/length(indp)
fp_test                    = 1 - sum(yest_test(indn) == y(indn))/length(indn)
Perf_test                  = sum(yest_test == y)/length(y)
[tpp_test , fpp_test]      = basicroc(y , fx_test);

[dum , ind]                = sort(y , 'descend');
figure
plot(fx_test(ind))
title(sprintf('Output of the strong classifier for test data with T = %d' , options.T))

figure
plot(fpp_train , tpp_train , fpp_test , tpp_test , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
grid on
title(sprintf('ROC for MBLBP features with T = %d' , options.T))
legend('Train' , 'Test' , 'Location' , 'SouthEast')


disp('Press key to continue')
pause


%% Fifth example : Adaboosting versus Gentleboosting with MLBP


load viola_24x24

y                          = int8(y);
indp                       = find(y == 1);
indn                       = find(y ==-1);

Ny                         = 24;
Nx                         = 24;
N                          = 8;
R                          = 1;

options.F                  = mblbp_featlist(Ny , Nx);
% mapping                    = getmapping(N,'u2');
% options.map                = uint8(mapping.table);
options.map                = uint8(0:255);

options.T                  = 10;

H                          = mblbp(X , options);


figure
imagesc(H)
title('MBLBP Features')
drawnow

index                      = randperm(length(y));  %shuffle data to avoid numerical discrepancies with long sequence of same label
tic,model0.param           = mblbp_gentleboost_binary_train_cascade(H(: , index) , y(index) , options.T);,toc
[yest0_train , fx0_train]  = mblbp_gentleboost_binary_predict_cascade(H , model0);

tp0_train                  = sum(yest0_train(indp) == y(indp))/length(indp)
fp0_train                  = 1 - sum(yest0_train(indn) == y(indn))/length(indn)
Perf0_train                = sum(yest0_train == y)/length(y)


tic,model1.param           = mblbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options.T);,toc
[yest1_train , fx1_train]  = mblbp_adaboost_binary_predict_cascade(H , model1);

tp1_train                  = sum(yest1_train(indp) == y(indp))/length(indp)
fp1_train                  = 1 - sum(yest1_train(indn) == y(indn))/length(indn)
Perf1_train                = sum(yest1_train == y)/length(y)


[dum , ind]               = sort(y , 'descend');

figure
plot((1:length(y)) , fx0_train(ind) , (1:length(y)) , fx1_train(ind) , 'r')
title(sprintf('Output of the strong classifier for train data with T = %d' , options.T))
legend('Gentleboost' , 'Adaboost')


[tpp0_train , fpp0_train]  = basicroc(y , fx0_train);
[tpp1_train , fpp1_train]  = basicroc(y , fx1_train);


load jensen_24x24

y                          = int8(y);
indp                       = find(y == 1);
indn                       = find(y ==-1);


H                          = mblbp(X , options);

[yest0_test , fx0_test]    = mblbp_gentleboost_binary_predict_cascade(H , model0);
[yest1_test , fx1_test]    = mblbp_gentleboost_binary_predict_cascade(H , model1);


tp0_test                   = sum(yest0_test(indp) == y(indp))/length(indp)
fp0_test                   = 1 - sum(yest0_test(indn) == y(indn))/length(indn)
Perf0_test                 = sum(yest0_test == y)/length(y)


tp1_test                   = sum(yest1_test(indp) == y(indp))/length(indp)
fp1_test                   = 1 - sum(yest1_test(indn) == y(indn))/length(indn)
Perf1_test                 = sum(yest1_test == y)/length(y)


[dum , ind]                = sort(y , 'descend');

figure
plot((1:length(y)) , fx0_test(ind) , (1:length(y)) , fx1_test(ind) , 'r')
title(sprintf('Output of the strong classifier for test data with T = %d' , options.T))
legend('Gentleboost' , 'Adaboost')


[tpp0_test , fpp0_test]   = basicroc(y , fx0_test);
[tpp1_test , fpp1_test]   = basicroc(y , fx1_test);


figure
plot(fpp0_train , tpp0_train , 'b--' , fpp0_test , tpp0_test , 'b' , fpp1_train , tpp1_train , 'r--' , fpp1_test , tpp1_test , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
grid on
title(sprintf('ROC for Gentleboosting & Adaboosting with MBLBP features and T = %d' , options.T))
legend('Train Gentleboost' , 'Test Gentleboost'  , 'Train Adaboost' , 'Test Adaboost' , 'Location' , 'SouthEast' )


disp('Press key to continue')
pause



%% Sixth example : MLBP applied on Viola-Jones database

load viola_24x24

[Ny,Nx,P]         = size(X);
N                 = 8;
scale             = 2*[1  ; 1 ];

% mapping   = getmapping(N,'u2');
% options.map     = uint8(mapping.table);
options.map       = uint8(0:2^N-1);
options.F         = mblbp_featlist(Ny , Nx , scale);
z                 = mblbp(X , options);

template          = options.F(: , 1);

Nxx               = (Nx-3*template(4) + 1);
Nyy               = (Ny-3*template(5) + 1);

Xmlbp             = zeros(Ny , Nx , P , class(z));
for i = 1:P
    I                        = reshape(z(: , i) , [Nyy , Nxx]);
    Xmlbp(: , : , i)         = imresize(I , [Ny , Nx]);
end

figure
display_database(X);
title(sprintf('Original database (click zoom to see images)'));

figure
display_database(Xmlbp);
title(sprintf('MLBP''s features with scale %d (click zoom to see images)' , scale(1)));



%% Seventh example : MLBP applied on Viola-Jones database

load viola_24x24
load model_detector_mblbp_24x24_4

bestFeat          = model.param(1 , 1); %40478+1;
thresh            = model.param(2 , 1);

[Ny,Nx,P]         = size(X);
N                 = 8;
options.map       = uint8(0:2^N-1);


FF                = mblbp_featlist(Ny , Nx);
options.F         = FF(: , bestFeat);
z                 = mblbp(X , options);

indpos            = find(y==1);
indneg            = find(y==-1);

Pr_pos            = sum(z(indpos)>=thresh)/length(indpos);
Pr_neg            = sum(z(indneg)<thresh)/length(indneg);


figure
plot(indpos , z(indpos) , indneg , z(indneg) , 'r' , (1:length(z)) , thresh*ones(1,length(z)) , 'g')
legend('Faces' , 'Non-faces' , '\theta')
title(sprintf('Best MBLBPr Feature = %d' , bestFeat))

figure
[Nneg , Xneg] = hist(double(z(indneg)) , 100 , 'r' );
bar(Xneg , Nneg)
set(get(gca , 'children') , 'facecolor' , [1 0 1])

hold on
[Npos , Xpos] = hist(double(z(indpos)) , 100 );
bar(Xpos , Npos);
plot(thresh*ones(1,2) , [0 , 1.2*max([Nneg , Npos])] , 'g' , 'linewidth' , 2)
hold off
legend(get(gca , 'children') , '\theta' , sprintf('Faces, Pr(z>=\\theta|y=1)=%4.2f' , Pr_pos) , sprintf('Non-faces, Pr(z<\\theta|y=-1)=%4.2f' , Pr_neg) , 'location' , 'Northwest' )
axis([-1 , 256 , 0 , 1.2*max([Nneg , Npos])])
title(sprintf('Best MBLBP Feature = %d' , bestFeat))

disp('Press key to continue')
pause


%% Eight example : Comparaison between full Strong classifier and cascade's technics


load('model_detector_mblbp_24x24_4');   %model trained on Viola-Jones database %
load jensen_24x24

options.param         = model.param;
options.map           = model.map;
options.F             = model.F;
options.dimsItraining = model.dimsItraining;
options.T             = size(model.param , 2);

%cascade              = [15 , 15 , 29 ; -0.75 ,  -0.25 , 0 ];
%cascade              = [3 , 7 , 10 , 20 , 25 ; -0.25 , -0*0 , 0*0.25 ,  0 , 0 ];


thresh               = 0;

indp                 = find(y == 1);
indn                 = find(y ==-1);

options.cascade_type = 0;
options.cascade      = [4 , 6 , 10 , 15 , 15 , 20 , 30 ; -1 , -0*75 , -0.5 ,  -0.25 , 0 , 0 , 0 ];
tic,fx_cascade       = eval_mblbp(X , options);,toc
yest                 = int8(sign(fx_cascade));


tp                   = sum(yest(indp) == y(indp))/length(indp)
fp                   = 1 - sum(yest(indn) == y(indn))/length(indn)
perf                 = sum(yest == y)/length(y)
[tpp1 , fpp1 ]       = basicroc(y , fx_cascade);



options.cascade_type = 1;
options.cascade      = [2 , 8 , 10 , 15 , 15 , 20 , 30 ; -0.5 , -0.25 , 0 , 0 , 0 , 0 , 0];

tic,fx_multiexit     = eval_mblbp(X , options);,toc
yest                 = int8(sign(fx_multiexit));


tp                   = sum(yest(indp) == y(indp))/length(indp)
fp                   = 1 - sum(yest(indn) == y(indn))/length(indn)
perf                 = sum(yest == y)/length(y)
[tpp2 , fpp2 ]       = basicroc(y , fx_multiexit);


options.cascade      = [length(options.param) ; 0];
tic,fx               = eval_mblbp(X , options);,toc
yest                 = int8(sign(fx - thresh));


tp                   = sum(yest(indp) == y(indp))/length(indp)
fp                   = 1 - sum(yest(indn) == y(indn))/length(indn)
perf                 = sum(yest == y)/length(y)
[tpp3 , fpp3 ]       = basicroc(y , fx);


figure
plot(1:length(y) , fx , 'r' , 1:length(y) , fx_cascade , 'b' ,  1:length(y) , fx_multiexit , 'k')

figure
plot(fpp1 , tpp1 , fpp2 , tpp2 , 'k' , fpp3 , tpp3 , 'r' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
legend('Cascade' , 'MultiExit', 'Full', 'Location' , 'SouthEast')
grid on
title(sprintf('ROC for Gentleboosting with T = %d for different technics of cascading' , options.T))


%% Ninth example : Comparaison between MBLBP and Extented MBLBP

load viola_24x24
[Ny , Nx , N]                       = size(X);

y                                   = int8(y);
indp                                = find(y == 1);
indn                                = find(y == -1);

options.T                           = 10;
options.F                           = mblbp_featlist(Ny , Nx);

% mapping                            = getmapping(N,'u2');
% options.map                        = uint8(mapping.table);
options.map                         = uint8(0:255);


H                                   = mblbp(X , options);
figure
imagesc(H)
title('MBLBP Features')
drawnow

index                               = randperm(length(y));  %shuffle data to avoid numerical discrepancies with long sequence of same label


model0.param                        = mblbp_adaboost_binary_train_cascade(H(: , index) , y(index) , options);
[yest_train , fx_train]             = mblbp_adaboost_binary_predict_cascade(H ,model0);

tp_train                            = sum(yest_train(indp) == y(indp))/length(indp)
fp_train                            = 1 - sum(yest_train(indn) == y(indn))/length(indn)
Perf_train                          = sum(yest_train == y)/length(y)
[tpp_train , fpp_train]             = basicroc(y , fx_train);


Xgrad                               = zeros(Ny , Nx , N , 'uint8');
for i = 1 : N
    [fx,fy]                         = gradient(double(X(: , : , i)));
    Igrad                           = sqrt((fx.*fx + fy.*fy));
    Xgrad(: , : , i)                = d2uint8_image(Igrad);
end


Hgrad                                = mblbp(Xgrad , options);

figure
imagesc(Hgrad)
title('E-MBLBP Features')
drawnow


model1.param                         = mblbp_adaboost_binary_train_cascade(Hgrad(: , index) , y(index) , options);
[yest_grad_train , fx_grad_train]    = mblbp_adaboost_binary_predict_cascade(Hgrad , model1);

tp_grad_train                        = sum(yest_grad_train(indp) == y(indp))/length(indp)
fp_grad_train                        = 1 - sum(yest_grad_train(indn) == y(indn))/length(indn)
Perf_grad_train                      = sum(yest_grad_train == y)/length(y)
[tpp_grad_train , fpp_grad_train]    = basicroc(y , fx_grad_train);


[dum , ind]                          = sort(y , 'descend');
figure
plot(1:length(y) , fx_train(ind) , 1:length(y) , fx_grad_train(ind) , 'r')
title(sprintf('Output of the strong classifier for train data with T = %d' , options.T))
legend('MBLBP' , 'E-MBLBP')


load jensen_24x24
[Ny , Nx , N]                       = size(X);

y                                   = int8(y);
indp                                = find(y == 1);
indn                                = find(y ==-1);

H                                   = mblbp(X , options);
[yest_test , fx_test]               = mblbp_adaboost_binary_predict_cascade(H , model0);

tp_test                             = sum(yest_test(indp) == y(indp))/length(indp)
fp_test                             = 1 - sum(yest_test(indn) == y(indn))/length(indn)
Perf_test                           = sum(yest_test == y)/length(y)
[tpp_test , fpp_test]               = basicroc(y , fx_test);

Xgrad                               = zeros(Ny , Nx , N , 'uint8');
for i = 1 : N
    [fx,fy]                         = gradient(double(X(: , : , i)));
    Igrad                           = sqrt((fx.*fx + fy.*fy));
    Xgrad(: , : , i)                = d2uint8_image(Igrad);
end

Hgrad                                = mblbp(Xgrad , options);
[yest_grad_test , fx_grad_test]      = mblbp_adaboost_binary_predict_cascade(Hgrad , model1);

tp_grad_test                         = sum(yest_grad_test(indp) == y(indp))/length(indp)
fp_grad_test                         = 1 - sum(yest_grad_test(indn) == y(indn))/length(indn)
Perf_grad_test                       = sum(yest_grad_test == y)/length(y)
[tpp_grad_test , fpp_grad_test]      = basicroc(y , fx_grad_test);


[dum , ind]                          = sort(y , 'descend');
figure
plot(1:length(y) , fx_test(ind) , 1:length(y) , fx_grad_test(ind) ,'r')
title(sprintf('Output of the strong classifier for test data with T = %d' , options.T))
legend('MBLBP' , 'E-MBLBP')


figure
plot(fpp_train , tpp_train , fpp_grad_train , tpp_grad_train , 'b.-' , fpp_test , tpp_test , 'r' , fpp_grad_test , tpp_grad_test , 'r.-' , 'linewidth' , 2)
axis([-0.02 , 1.02 , -0.02 , 1.02])
title(sprintf('ROC for MBLBP features with T = %d' , options.T))
legend('Train MBLBP' , 'Train E-MBLBP' , 'Test MBLBP' , 'Test E-MBLBP' , 'Location' , 'SouthEast')


disp('Press key to continue')
pause

