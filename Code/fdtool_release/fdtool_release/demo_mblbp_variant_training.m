%% Demo illustrating variant of MBLBP features


clc,clear, close all, drawnow
load viola_24x24
options                                = load('haar_dico_2.mat');
[Ny , Nx , P]                          = size(X);
y                                      = int8(y);

options.T                              = 30;
options.transpose                      = 1;
options.F                              = mblbp_featlist(Ny , Nx);


N                                      = 3000;
vect                                   = [1:N , 5001:5001+N-1];
indextrain                             = vect(randperm(length(vect)));
indextest                              = (1:length(y));
indextest(indextrain)                  = [];

ytrain                                 = y(indextrain);
ytest                                  = y(indextest);

indp                                   = find(ytest == 1);
indn                                   = find(ytest ==-1);




%% MBLBP  Adaboost with normal mapping%%

options.map                            = uint8(0:255);


Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpada1 , fxtest_mblbpada1]     = mblbp_adaboost_binary_predict_cascade(Htest , options);

tp_mblbpada1                            = sum(ytest_mblbpada1(indp) == ytest(indp))/length(indp);
fp_mblbpada1                            = 1 - sum(ytest_mblbpada1(indn) == ytest(indn))/length(indn);
perf_mblbpada1                          = sum(ytest_mblbpada1 == ytest)/length(ytest);

[tpp_mblbpada1 , fpp_mblbpada1]        = basicroc(ytest , fxtest_mblbpada1);

%% MBLBP  Adaboost with Rotation independant mapping%%

mapping                                = getmapping(8,'ri');
options.map                            = uint8(mapping.table);

Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpada2 , fxtest_mblbpada2]   = mblbp_adaboost_binary_predict_cascade(Htest , options);

tp_mblbpada2                            = sum(ytest_mblbpada2(indp) == ytest(indp))/length(indp);
fp_mblbpada2                            = 1 - sum(ytest_mblbpada2(indn) == ytest(indn))/length(indn);
perf_mblbpada2                          = sum(ytest_mblbpada2 == ytest)/length(ytest);

[tpp_mblbpada2 , fpp_mblbpada2]        = basicroc(ytest , fxtest_mblbpada2);


%% MBLBP  Adaboost with uniform mapping%%

mapping                                = getmapping(8,'u2');
options.map                            = uint8(mapping.table);

Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpada3 , fxtest_mblbpada3]   = mblbp_adaboost_binary_predict_cascade(Htest , options);

tp_mblbpada3                            = sum(ytest_mblbpada3(indp) == ytest(indp))/length(indp);
fp_mblbpada3                            = 1 - sum(ytest_mblbpada3(indn) == ytest(indn))/length(indn);
perf_mblbpada3                          = sum(ytest_mblbpada3 == ytest)/length(ytest);

[tpp_mblbpada3 , fpp_mblbpada3]        = basicroc(ytest , fxtest_mblbpada3);



%% CS-MBLBP  Adaboost %%

mapping                                = getmapping(8,'u2');
options.map                            = uint8(0:255);
options.cs_opt                         = 1;

Htrain                                 = mblbp(X(:,:,indextrain) , options);
if(options.transpose)
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
else
    tic,options.param                  = mblbp_adaboost_binary_train_cascade(Htrain , ytrain , options);,toc
end
clear Htrain
Htest                                  = mblbp(X(:,:,indextest) , options);
[ytest_mblbpada4 , fxtest_mblbpada4]   = mblbp_adaboost_binary_predict_cascade(Htest , options);

tp_mblbpada4                            = sum(ytest_mblbpada4(indp) == ytest(indp))/length(indp);
fp_mblbpada4                            = 1 - sum(ytest_mblbpada4(indn) == ytest(indn))/length(indn);
perf_mblbpada4                          = sum(ytest_mblbpada4 == ytest)/length(ytest);

[tpp_mblbpada4 , fpp_mblbpada4]        = basicroc(ytest , fxtest_mblbpada4);



figure(1)
plot(fpp_mblbpada1 , tpp_mblbpada1  , 'b', fpp_mblbpada2 , tpp_mblbpada2  , 'k',  fpp_mblbpada3 , tpp_mblbpada3 , 'r' , fpp_mblbpada4 , tpp_mblbpada4 , 'c' , 'linewidth' , 2)
grid on
legend('LBP' , 'LBP-ri' , 'LBP-u2' , 'CS-LBP' , 'location' , 'southeast')
title(sprintf('Performances for 4 variants of LBP'))
axis([-0.02 , 1.02 , -0.02 , 1.02])
