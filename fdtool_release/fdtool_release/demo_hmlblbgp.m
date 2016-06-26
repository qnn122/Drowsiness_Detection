% HMBLBP vs HMBLGP


clc,clear, close all, drawnow
load viola_24x24
Xviola                                 = X;
yviola                                 = y;
load jensen_24x24
X                                      = cat(3 , X , Xviola);
y                                      = [y , yviola];
[Ny , Nx , P]                          = size(X);
y                                      = int8(y);


N                                      = 1000;


%options.spyr                           = [1 , 1 , 1 , 1 , 1];
%options.spyr                           = [1 , 1 , 1 , 1 , 1 ; 1/4 , 1/4 , 1/4 , 1/4 , 1/16];
options.spyr                           = [1 , 1 , 1 , 1, 1];
options.scale                          = [1];
options.cs_opt                         = 0;
options.maptable                       = 0;
options.useFF                          = 0;
options.improvedLBP                    = 0;
options.improvedLGP                    = 0;
options.rmextremebins                  = 1;
options.color                          = 0;
options.norm                           = [0,0,0];
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

vect                                   = [1:N , 9916+1:9916+1+N-1];
%indextrain                             = vect(randperm(length(vect)));
indextrain                             = vect;
indextest                              = (1:length(y));
indextest(indextrain)                  = [];

ytrain                                 = y(indextrain);
ytest                                  = y(indextest);

indp                                   = find(ytrain == 1);
indn                                   = find(ytrain ==-1);




% mlhmslbp_spyr  Features
d                                      = options.nbin*options.nH*options.nscale*options.ncolor;
Hmblbp                                 = zeros(d , length(indextrain));
Hmblgp                                 = zeros(d , length(indextrain));

for i = 1 : length(indextrain)
    %   Htrain(: , i)                       = mlhmslbp_spyr(X(:,:,indextrain(i)) , options);
    [fxtemp , ytemp , Hmblbp(: , i)]    = eval_hmblbp_spyr_subwindow(X(:,:,indextrain(i)) , options);
    [fxtemp , ytemp , Hmblgp(: , i)]    = eval_hmblgp_spyr_subwindow(X(:,:,indextrain(i)) , options);
end


Hmblbp_p = mean(Hmblbp( : , 1:N) , 2);
Hmblbp_n = mean(Hmblbp( : , N+1:2*N) , 2);
Hmblgp_p = mean(Hmblgp( : , 1:N) , 2);
Hmblgp_n = mean(Hmblgp( : , N+1:2*N) , 2);




Pmblbp  = sum((Hmblbp_p - Hmblbp_n).^2)
Pmblgp  = sum((Hmblgp_p - Hmblgp_n).^2)


figure(1)
plot(1:options.nbin , Hmblbp_p , 1:options.nbin , Hmblbp_n)
h=legend('H_k(B^f)' , 'H_k(B^{nf})');
axis([-1 , options.nbin+2 , -1 , max([Hmblbp_p ; Hmblbp_n])+1])
title(sprintf('P_{LBP} = %5.5f' , Pmblbp));
%set(h , 'interpreter' , 'latex');

figure(2)
plot(1:options.nbin , Hmblgp_p , 1:options.nbin , Hmblgp_n)
h=legend('H_k(G^f)' , 'H_k(G^{nf})');
axis([-1 , options.nbin+2 , -1 , max([Hmblgp_p ; Hmblgp_n])+1])
title(sprintf('P_{LGP} = %5.5f' , Pmblgp));

drawnow

ON        = ones(1 , N);
dmblbp_pp = zeros(N , N);
dmblbp_pn = zeros(N , N);
dmblgp_pp = zeros(N , N);
dmblgp_pn = zeros(N , N);


for i = 1:N
    
    Hilbp_p          = Hmblbp(: , i);
    Hmblbptemp       = Hilbp_p(: , ON);
    dmblbp_pp(i , :) = sum((Hmblbptemp - Hmblbp( : , 1:N)).^2 , 1);
    dmblbp_pn(i , :) = sum((Hmblbptemp - Hmblbp( : , N+1:2*N)).^2 , 1);
    
    
    Hilgp_p          = Hmblgp(: , i);
    Hmblgptemp       = Hilgp_p(: , ON);
    dmblgp_pp(i , :) = sum((Hmblgptemp - Hmblgp( : , 1:N)).^2 , 1);
    dmblgp_pn(i , :) = sum((Hmblgptemp - Hmblgp( : , N+1:2*N)).^2 , 1);
    
    i
    
end

Nbins = 200;

maxi = max([max(dmblbp_pp(:)) , max(dmblbp_pn(:)) , max(dmblgp_pp(:)) , max(dmblgp_pn(:))])/3;
mini = min([min(dmblbp_pp(:)) , min(dmblbp_pn(:)) , min(dmblgp_pp(:)) , min(dmblgp_pn(:))]);

vect = (mini : (maxi-mini)/(Nbins - 1) : maxi);

Nmblbp_pp = histc(dmblbp_pp(:) , vect);
Nmblbp_pn = histc(dmblbp_pn(:) , vect);

Nmblgp_pp = histc(dmblgp_pp(:) , vect);
Nmblgp_pn = histc(dmblgp_pn(:) , vect);


figure(3)
bar(vect , [Nmblbp_pp , Nmblbp_pn])
axis([mini - 2 , maxi+2 , 0 , max([Nmblbp_pp ; Nmblbp_pn]) + 20])
legend('d_{LBP}^{f,f}' , 'd_{LBP}^{f,nf}')
xlabel('d')
ylabel('#')

figure(4)

bar(vect , [Nmblgp_pp , Nmblgp_pn])
axis([mini - 2 , maxi+2 , 0 , max([Nmblgp_pp ; Nmblgp_pn]) + 20])
legend('d_{LGP}^{f,f}' , 'd_{LGP}^{f,nf}')
xlabel('d')
ylabel('#')





