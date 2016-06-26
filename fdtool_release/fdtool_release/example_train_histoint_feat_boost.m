close all
options.positives_path     = fullfile(pwd , 'images' , 'test' , 'positives');
options.negatives_path     = fullfile(pwd , 'images' , 'train' , 'negatives');
options.posext             = {'pgm'};
options.negext             = {'jpg'};
options.negmax_size        = 1200;
options.preview            = 0;
options.Npostrain          = 9000;
options.Nnegtrain          = 15000;
options.Npostest           = 3000;
options.Nnegtest           = 10000;
options.Nposboost          = 0;
options.Nnegboost          = 1000;
options.boost_ite          = 5;
options.seed               = 5489;
options.resetseed          = 1;
options.probaflipIpos      = 0.5;
options.probarotIpos       = 0.01;
options.m_angle            = 0;
options.sigma_angle        = 2^2;
options.probaswitchIneg    = 0.9;
options.posscalemin        = 0.8;
options.posscalemax        = 1.5;
options.negscalemin        = 0.7;
options.negscalemax        = 2.5;
options.typefeat           = 3;
options.addbias            = 1;
options.max_detections     = 5000;
options.num_threads        = -1;
options.dimsIscan          = [24 , 24];
options.scalingbox         = [2 , 1.4 , 1.8];
options.mergingbox         = [1/2 , 1/2 , 1/3];
options.spyr               = [1 , 1 , 1 , 1 , 1]; %[1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4];
options.scale              = [1];
options.maptable           = 0;
options.cs_opt             = 0;
options.improvedLBP        = 0;
options.rmextremebins      = 0;
options.color              = 0;
options.norm               = [0 , 0 , 4]; %[0 , 0 , 2]
options.clamp              = 0.2;
options.n                  = 0;
options.L                  = 1.2;
options.kerneltype         = 0;
options.numsubdiv          = 8;
options.minexponent        = -20;
options.maxexponent        = 8;
options.s                  = 2;
options.B                  = 1;
options.c                  = 0.1; %2
[options , model]          = train_histoint_feat_boost(options);

save model_spyr_s1_spyr_1_R8 model

figure(1)
plot(options.fpp , options.tpp  , 'b', 'linewidth' , 2)
grid on
title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , options.perftest , options.auc_est))
axis([-0.02 , 1.02 , -0.02 , 1.02])

figure(2)
semilogy(1:options.boost_ite , options.pd_per_stage , 1:options.boost_ite , options.pfa_per_stage , 'linewidth' , 2)
legend('P_d' , 'P_{fa}' , 'location' , 'southwest')





options.positives_path     = fullfile(pwd , 'images' , 'train' , 'positives');
options.negatives_path     = fullfile(pwd , 'images' , 'test' , 'negatives');
options.posext             = {'png'};
options.negext             = {'jpg'};
options.negmax_size        = 1200;
options.preview            = 0;
options.Npos               = 3000;
options.Nneg               = 5000;
options.probaflipIpos      = 0.5;
options.probarotIpos       = 0.00;
options.m_angle            = 0;
options.sigma_angle        = 5^2;
options.seed               = 5489;
options.resetseed          = 1;
options.probaswitchIneg    = 0.5;
options.posscalemin        = 0.7;
options.posscalemax        = 1.3;
options.negscalemin        = 0.7;
options.negscalemax        = 3;
options.typefeat           = 3;


[fx , y]                   = eval_model_dataset(options , model);
yest                       = sign(fx);
accuracy                   = sum(yest==y)/length(y);
[tpp , fpp]                = basicroc(y , fx);
auc_est                    = auroc(tpp', fpp');


save fpp_tpp_hmslbp_spyr1_s1_R8 fpp tpp fx y

figure(3)
plot(fpp , tpp  , 'b', 'linewidth' , 2)
grid on
title(sprintf('Accuracy = %4.3f, AUC = %4.3f' , accuracy , auc_est))
axis([-0.02 , 1.02 , -0.02 , 1.02])

figure(4)
plot(fx)


% figure(2)
% plot(model.w)

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