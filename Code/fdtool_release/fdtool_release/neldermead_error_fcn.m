
function [xsol , fsol , func_evals] = neldermead_error_fcn(funfcn , x , varargin)


tolx            = 10e-6;
tolf            = 10e-6;
maxfun          = 200;
maxiter         = 200;

rho             = 1;
chi             = 2;
psi             = 0.5;
sigma           = 0.5;

v0              = x;    % Place input guess in the simplex! (credit L.Pfeffer at Stanford)
fv0             = funfcn(v0 , varargin{:});

func_evals      = 1;
itercount       = 0;

usual_delta     = 0.05;             % 5 percent deltas for non-zero terms
zero_term_delta = 0.00025;      % Even smaller delta for zero elements of x

v1              = x;
if (v1 ~= 0)
    v1          = (1 + usual_delta)*v1;
else
    v1          = zero_term_delta;
end
fv1             = funfcn(v1,varargin{:});
if(fv1 < fv0)
    temp        = fv1;
    fv1         = fv0;
    fv0         = temp;
    
    temp        = v1;
    v1          = v0;
    v0          = temp;
end

how             = 0;


itercount       = itercount + 1;
func_evals      = func_evals+1;

while (func_evals < maxfun && itercount < maxiter)
    if ( (abs(fv0-fv1) <= max(tolf,10*eps(fv0))) && (abs(v1-v0) <= max(tolx,10*eps(v0))) )
        break
    end
    
    xbar       = v0;
    xr         = (1.0 + rho)*xbar - rho*v1;
    fxr        = funfcn(xr,varargin{:});
    func_evals = func_evals+1;
    
    if (fxr < fv0)
        xe         = (1 + rho*chi)*xbar - rho*chi*v1;
        fxe        = funfcn(xe,varargin{:});
        func_evals = func_evals+1;
        if (fxe < fxr)
            v1  = xe;
            fv1 = fxe;
            how = 1;
        else
            v1  = xr;
            fv1 = fxr;
            how = 2;
        end
    else
        % Perform contraction
        if (fxr < fv1)
            % Perform an outside contraction
            xc         = (1 + psi*rho)*xbar - psi*rho*v1;
            fxc        = funfcn(xc,varargin{:});
            func_evals = func_evals+1;
            if (fxc <= fxr)
                v1     = xc;
                fv1    = fxc;
                how    = 3;
            else
                % perform a shrink
                how    = 4;
            end
        else
            % Perform an inside contraction
            xcc         = (1-psi)*xbar + psi*v1;
            fxcc        = funfcn(xcc,varargin{:});
            func_evals  = func_evals+1;       
            if (fxcc < fv1)
                v1     = xcc;
                fv1    = fxcc;
                how    = 3;
            else
                % perform a shrink
                how    = 4;
            end
        end
        if (how == 4)
            v1         = v0 + sigma*(v1 - v0);
            fv1        = funfcn(v1,varargin{:});
            func_evals = func_evals + 1;
        end
    end
    
    if(fv1 < fv0)
        temp        = fv1;
        fv1         = fv0;
        fv0         = temp;
        
        temp        = v1;
        v1          = v0;
        v0          = temp;
    end
    
    itercount = itercount + 1;
    
end

xsol    = v0;
fsol    = fv0;






