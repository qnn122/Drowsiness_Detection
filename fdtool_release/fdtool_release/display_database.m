function display_database(X)

%  Display database image on a single figure 
%
%  Usage
%  ------
% 
%  display_database(X)
%
%  Input
%  ------
%   
%  X                 Image's database (Ny x Nx x N) in UINT8 format
% 
%
%  Example
%  -------
%
%  load viola_24x24
%  display_database(X)
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%

[Ny , Nx , N] = size(X);

c             = class(X);

P             = ceil(sqrt(N));

Y             = permute(cat(3 , X , zeros(Ny , Nx , P*P-N , c)) , [1 , 3 , 2]);

Y             = reshape(Y , [Ny , P , P , Nx]);

Y             = reshape(permute(Y , [1 , 2 , 4 , 3]) , Ny*P , Nx*P);

imagesc(Y)
colormap(gray)
axis off



