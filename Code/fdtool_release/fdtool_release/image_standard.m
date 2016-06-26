function I = image_standard(X)

[Ny , Nx , N]      = size(X);

X                  = reshape(double(X) , Ny*Nx , N);
I                  = reshape(X./repmat(std(X , 1 , 1) , Ny*Nx , 1) , [Ny , Nx , N ]);

