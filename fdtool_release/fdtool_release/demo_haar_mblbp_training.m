%% Demo illustrating Performances of different boosting method for both
%% Harr & MLBP features 



clc,clear, close all, drawnow
load viola_24x24
options                                = load('haar_dico_2.mat');
II                                     = image_integral_standard(X);
y                                      = int8(y);
options.T                              = 64;
options.transpose                      = 1;
options.usesingle                      = 1;


[Ny , Nx , P]                          = size(II);
options.F                              = haar_featlist(Ny , Nx , options.rect_param);
options.G                              = Haar_matG(Ny , Nx , options.rect_param);


N                                      = 3000;
vect                                   = [1:N , 5001:5001+N-1];
indextrain                             = vect(randperm(length(vect)));
indextest                              = (1:length(y));
indextest(indextrain)                  = [];

ytrain                                 = y(indextrain);
ytest                                  = y(indextest);

indp                                   = find(ytest == 1);
indn                                   = find(ytest ==-1);

%% Fast Haar Adaboost %%
options.fine_threshold                 = 1;


tic,options.param                      = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
[ytest_fhaarada , fxtest_fhaarada]     = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);


tp_fhaarada                            = sum(ytest_fhaarada(indp) == ytest(indp))/length(indp);
fp_fhaarada                            = 1 - sum(ytest_fhaarada(indn) == ytest(indn))/length(indn);
perf_fhaaradawf                        = sum(ytest_fhaarada == ytest)/length(ytest);

[tpp_fhaaradawf , fpp_fhaaradawf]      = basicroc(ytest , fxtest_fhaarada);


options.fine_threshold                 = 0;


tic,options.param                      = fast_haar_adaboost_binary_train_cascade(II(: , : , indextrain) , ytrain , options);,toc
[ytest_fhaarada , fxtest_fhaarada]     = haar_adaboost_binary_predict_cascade(II(: , : , indextest) , options);


tp_fhaarada                            = sum(ytest_fhaarada(indp) == ytest(indp))/length(indp);
fp_fhaarada                            = 1 - sum(ytest_fhaarada(indn) == ytest(indn))/length(indn);
perf_fhaaradawof                       = sum(ytest_fhaarada == ytest)/length(ytest);

[tpp_fhaaradawof , fpp_fhaaradawof]    = basicroc(ytest , fxtest_fhaarada);



% %%  Haar Gentleboost %%

Htrain                                 = haar(X(:,:,indextrain) , options );

if(options.transpose)
    tic,options.param                    = haar_gentleboost_binary_train_cascade_memory(Htrain , ytrain , options);,toc
else
    tic,options.param                    = haar_gentleboost_binary_train_cascade_memory(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = haar(X(:,:,indextest) , options);
[ytest_haargen , fxtest_haargen]       = haar_gentleboost_binary_predict_cascade_memory(Htest , options);

tp_haargen                             = sum(ytest_haargen(indp) == ytest(indp))/length(indp);
fp_haargen                             = 1 - sum(ytest_haargen(indn) == ytest(indn))/length(indn);
perf_haargen                           = sum(ytest_haargen == ytest)/length(ytest);

[tpp_haargen , fpp_haargen]            = basicroc(ytest , fxtest_haargen);

%%  Haar Adaboost %%

Htrain                                 = haar(X(:,:,indextrain) , options );

if(options.transpose)
    tic,options.param                  = haar_adaboost_binary_train_cascade_memory(Htrain , ytrain , options);,toc
else
    tic,options.param                  = haar_adaboost_binary_train_cascade_memory(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = haar(X(:,:,indextest) , options);
[ytest_haarada , fxtest_haarada]       = haar_gentleboost_binary_predict_cascade_memory(Htest , options);

tp_haarada                             = sum(ytest_haarada(indp) == ytest(indp))/length(indp);
fp_haarada                             = 1 - sum(ytest_haarada(indn) == ytest(indn))/length(indn);
perf_haarada                           = sum(ytest_haarada == ytest)/length(ytest);

[tpp_haarada , fpp_haarada]            = basicroc(ytest , fxtest_haarada);



%% MBLBP  Gentleboost %%

options.F                              = mblbp_featlist(Ny , Nx);
options.map                            = uint8(0:255);
Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_gentleboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_gentleboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpgen , fxtest_mblbpgen]     = mblbp_gentleboost_binary_predict_cascade(Htest , options);

tp_mblbpgen                            = sum(ytest_mblbpgen(indp) == ytest(indp))/length(indp);
fp_mblbpgen                            = 1 - sum(ytest_mblbpgen(indn) == ytest(indn))/length(indn);
perf_mblbpgen                          = sum(ytest_mblbpgen == ytest)/length(ytest);

[tpp_mblbpgen , fpp_mblbpgen]          = basicroc(ytest , fxtest_mblbpgen);

%% MBLBP  Adaboost %%


Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpada , fxtest_mblbpada]     = mblbp_adaboost_binary_predict_cascade(Htest , options);

tp_mblbpada                            = sum(ytest_mblbpada(indp) == ytest(indp))/length(indp);
fp_mblbpada                            = 1 - sum(ytest_mblbpada(indn) == ytest(indn))/length(indn);
perf_mblbpada                          = sum(ytest_mblbpada == ytest)/length(ytest);

[tpp_mblbpada , fpp_mblbpada]          = basicroc(ytest , fxtest_mblbpada);


figure(1)
plot(fpp_fhaaradawf , tpp_fhaaradawf , fpp_fhaaradawof , tpp_fhaaradawof , 'k' , fpp_haargen , tpp_haargen , 'g',  fpp_haarada , tpp_haarada , 'c' , fpp_mblbpgen , tpp_mblbpgen , 'r' , fpp_mblbpada , tpp_mblbpada  , 'm', 'linewidth' , 2)
grid on
legend('FastHaarAda with threshold' , 'FastHaarAda without threshold' , 'Haar Gentle' , 'Haar Ada' , 'MBLBP Gentle' , 'MBLBP Ada' , 'location' , 'southeast');
axis([-0.02 , 1.02 , -0.02 , 1.02])
title(sprintf('Performances for Haar and MBLBP features for different boosting algorithms with T = %d weaklearners' , options.T));
