

load Ineg

load model_detector_chlbp_24x24.mat


% N                          = [8 , 4];
% R                          = [1 , 1];
% map                        = zeros(2^max(N) , length(N));
% map(1:2^N(1) , 1)          = (0:2^N(1)-1)';%mapping.table';
% map(1:2^N(2) , 2)          = (0:2^N(2)-1)';
model.map                  = map;

% map                        = zeros(2^max(N) , length(N));
% mapping                    = getmapping(N(1),'u2');
% map(: , 1)                 = mapping.table';
% %map(: , 2)                 = mapping.table';
% 
% %map(1:2^N(2) , 2)          = (0:2^N(2)-1)';
% 
% model.map                  = map; 
%map                        = model.map;

shiftbox                   = cat(3 , [24 , 24 ; 1 , 1],[16 , 14 ; 4 , 4]);
%shiftbox                   = cat(3 , [24 , 24 ; 1 , 1]);


%model.cascade_type         = 0;

fx                         = eval_chlbp(Ineg , model)


H                          = chlbp(Ineg,N,R,map,shiftbox);

[~ , fx1]                  = chlbp_gentleboost_binary_predict_cascade(H,model.param)