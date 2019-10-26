function [out, gradz, hessian] = logdensityLogisticRegression(z, data, s2w)
% z    : (1 x n) latent variable
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

n = length(z);
Xz = data.X*z';
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(z*z')/s2w ...
      + sum( logsigmoid(data.titleY.*Xz) );
if nargout > 1
    % Evaluate gradient of the model
    sig = sigmoid(Xz);
    gradz  = - z/s2w + ...
             + (data.Y - sig)'*data.X;
    if nargout > 2 
       %hessian = - (1/s2w)*eye(n) - data.X'*(diag(sig.*(1-sig))*data.X);
       sqrtsigma = sqrt(sig.*(1-sig)); 
       sX = bsxfun(@times, sqrtsigma, data.X); 
       hessian = - (1/s2w)*eye(n) - sX'*sX;       
    end
end
