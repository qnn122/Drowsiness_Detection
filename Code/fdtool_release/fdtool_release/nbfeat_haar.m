function nF = nbfeat_haar(Ny , Nx , h , w)

% Return the total number of feature in a image of size (Ny x Nx) and a
% pattern of height h and weigth w
%
%  Usage
%  ------
% 
%  nF               = nbfeat_haar(Ny , Nx , h , w)
%
%  Inputs
%  -------
%   
%  Ny                Number of rows of the image's database
%  Nx                Number of columns of the image's database
%  h                 Pattern's height 
%  w                 Pattern's weight
% 
%  Outputs
%  -------
%
%  nF               Total number of Feature for each x,y position in
%                   [1,...,Ny]x[1,...,Nx], each scale s for given pattern's
%                   sizes
%
%  Examples
%  -------
%
%  nF = nbfeat_haar(24 , 24 , 2 , 1)
%  nF = nbfeat_haar(24 , 24 , [2 , 1 , 1 , 3 , 2] , [1 , 2 , 3 , 1 , 2] )
%  nF = nbfeat_haar(24 , 24 , [2 , 1] , [1 , 2 ] )
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009

if(nargin < 4)
    
    w  = [1 , 2];
    
end

if(nargin < 3)
    
    h  = [2 , 1];
    
end


if(nargin < 1)
    
    Ny  = 24;
    
end

if(nargin < 2)
    
    Nx  = Ny;
    
end

Y    = floor(Ny./h);
X    = floor(Nx./w);
nF   = sum(X.*Y.*((Nx+1) - w.*(X+1)./2).*((Ny+1) - h.*(Y+1)./2));