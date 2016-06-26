clc,clear, close all, drawnow
load viola_24x24
options                                = load('haar_dico_2.mat');
[Ny , Nx , P]                          = size(X);

I                                      = X(: , : , 1);

y                                      = 2;
x                                      = 6;
h                                      = 10;
w                                      = 10;

% y                                      = 1;
% x                                      = 1;
% h                                      = Ny;
% w                                      = Nx;


a                                      = y + (x - 1)*Ny;
b                                      = (y - 1 + h) + (x - 1 + w  - 1)*Ny;
c                                      = (y - 1) + (x - 1 + w  - 1)*Ny;
d                                      = (y - 1 + h) + (x - 1)*Ny;



Icrop                                  = I(y : y  + h - 1 , x: x  + w - 1); 


histo_direct                           = histc(Icrop(:) , (0:255))';



NyNx                                   = Ny*Nx;
vect_temp                              = (0:255)*NyNx;

R                                      = zeros(Ny , Nx , 256);
vect                                   = (1:NyNx)' + double((I(:) - 0))*NyNx;
R(vect)                                = 1;

% int_im                                 = cumsum(cumsum(R , 1) , 2);
% int_histo                              = (int_im(b + vect_temp) + (int_im(c + vect_temp) + int_im(d + vect_temp))  ) -  int_im(a + vect_temp);


%int_histo                              = (int_im(b + vect_temp) +  int_im(a + vect_temp) - (int_im(c + vect_temp) + int_im(d + vect_temp))  ) ;

int_histo                              = zeros(1 , 256);
for i = 1:256
    int_histo(i)                       = aire(uint8(R(: , : , i)) , y , x , h , w);
end


figure(1)
plot(0:255 , histo_direct , 0:255 , int_histo , 'r' , 'linewidth' , 2)

figure(2)

plot(0:255 , histo_direct/sum(histo_direct) , 0:255 , int_histo/sum(int_histo) , 'r' , 'linewidth' , 2)



