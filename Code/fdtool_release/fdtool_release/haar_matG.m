function G = haar_matG(Ny , Nx , rect_param)

% Construct the sparse matrix of Features for the FastAdaboost algorithm
%
%  Usage
%  ------
%
%  G = haar_matG([Ny] , [Nx] , [rect_param])
%
%  Inputs
%  -------
%
%  Ny                Number of rows of the image's database
%
%  Nx                Number of columns of the image's database
%
%  rect_param        Features rectangles parameters (10 x nR), where nR is the total number of rectangles for the patterns.
%                    (default Vertical(2 x 1) [1 ; -1] and Horizontal(1 x 2) [-1 , 1] patterns)
%                    rect_param(: , i) = [ip ; wp ; hp ; nrip ; nr ; xr ; yr ; wr ; hr ; sr], where
% 					 ip     index of the current pattern. ip = [1,...,nP], where nP is the total number of patterns
% 				     wp     width of the current pattern
% 					 hp     height of the current pattern
% 					 nrip   total number of rectangles for the current pattern ip
% 					 nr     index of the current rectangle of the current pattern, nr=[1,...,nrip]
% 					 xr,yr  top-left coordinates of the current rectangle of the current pattern
% 					 wr,hr  width and height of the current rectangle of the current pattern
% 					 sr     weights of the current rectangle of the current pattern
%
%  Output
%  ------
%
%  G                 Sparse Features matrix (Ny*Nx x nF) wher nF is the
%                    total number of Features for each pattern, x,y positions and scale factor
%                    (see nbfeat_haar function)
%
%  Example
%  -------
%
%  load haar_dico_2
%  Ny = 24;
%  Nx = 24;
%  G  = haar_matG(Ny , Nx , rect_param);
%
%  spy(G);
%  axis square;
%
%
%  Author : Sébastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 01/27/2009
%
% Use the int8tosparse mex file modified by J. Tursa

if(nargin < 3)
    
    rect_param = [1 1 2 2;1 1 2 2;2 2 1 1;2 2 2 2;1 2 1 2;0 0 0 1;0 1 0 0;1 1 1 1;1 1 1 1;1 -1 -1 1];
    
end

if(nargin < 1)
    
    Ny         = 24;
    
end

if(nargin < 2)
    
    Nx         = Ny;
    
end



[numR , indR]  = unique(rect_param(1 , :) , 'first');
nFeatures_type = length(numR);


invBt          = sparse(kron([zeros(Nx , 1) , -eye(Nx , Nx-1)] + eye(Nx) , [zeros(Ny , 1) , -eye(Ny , Ny-1)] + eye(Ny)));
nF             = nbfeat_haar(Ny , Nx , rect_param(3 , indR) , rect_param(2 , indR));
HH             = zeros(Ny*Nx , nF , 'int8');
ZZ             = zeros(Ny , Nx);
co             = 1;

for f = 1:nFeatures_type
    
    W     = rect_param(2 , indR(f));
    H     = rect_param(3 , indR(f));
    indRf = find(rect_param(1 , :) == numR(f));
    R     = length(indRf);
    
    for w = W : W : Nx
        for h = H:H:Ny
            for y=1:Ny-h+1
                for x=1:Nx-w+1
                    hh    = ZZ;
                    for r=1:R
                        
                        coeffw                      = w/rect_param(2 , indRf(r));
                        coeffh                      = h/rect_param(3 , indRf(r));
                        xr                          = x + coeffw*rect_param(6 , indRf(r));
                        yr                          = y + coeffh*rect_param(7 , indRf(r));
                        wr                          = coeffw*rect_param(8 , indRf(r));
                        hr                          = coeffh*rect_param(9 , indRf(r));
                        s                           = rect_param(10 , indRf(r));
                        hh(yr:yr+hr-1 , xr:xr+wr-1) = s;
                    end
                    
                    HH(: , co)                      = hh(:);
                    co                              = co+1;
                end
            end
        end
    end
end

G     = invBt*int8tosparse(HH);


