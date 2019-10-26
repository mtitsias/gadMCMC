function [out, gradz, hessian] = logdensityGaussian(x, mm, L, hessian)

diagL = 0;
if min(size(L))==1
  diagL = 1;
end

if diagL == 0
  aux = L\((x-mm)');
  out = -0.5*length(x)*log(2*pi) - sum(log(diag(L))) - 0.5*(aux'*aux);
else
  L2 = L.*L;  
  diff = x-mm;
  aux = (diff./L2)';
  out = -0.5*length(x)*log(2*pi) - sum(log(L)) - 0.5*(diff*aux);
end

if nargout > 1
  if diagL == 0  
     gradz = L'\aux;
     gradz = - gradz';
  else
     gradz = - aux';
  end
  if nargout > 2
    if nargin < 4  
    if diagL == 0   
       hessian = - L'\(L\eye(length(x)));
    else  
       hessian = - diag(1./L2); 
    end
    end
  end
end
