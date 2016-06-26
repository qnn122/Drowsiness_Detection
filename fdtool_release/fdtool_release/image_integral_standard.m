function II = image_integral_standard(X)

% Standardize and compute Image Integral 
%
%  Usage
%  ------
% 
%  II = image_integral_standard(X)
%
%  Input
%  ------
%   
%  X                 Image's database (Ny x Nx x N) in UINT8 format
% 
%  Output
%  ------
%
%  II                Image Integral (Ny x Nx x N) of standardized image I in DOUBLE format
%
%  Example
%  -------
%
%  load viola_24x24
%  II = image_integral_standard(X);
%  
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009

[Ny , Nx , N]      = size(X);

X                  = reshape(double(X) , Ny*Nx , N);
Xstandard          = reshape(X./repmat(std(X , 1 , 1) , Ny*Nx , 1) , [Ny , Nx , N ]);
II                 = cumsum(cumsum(Xstandard , 2));
