function [out, gradz, hessian] = logdensityLogisticRegression_nuts(z, data, s2w)
% z    : (1 x n) latent variable
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

z = z';
n = length(z);
Xz = data.X*z';
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(z*z')/s2w ...
      + sum( logsigmoid(data.titleY.*Xz) );
if nargout > 1
    % Evaluate gradient of the model
    sig = sigmoid(Xz);
    gradz  = - z/s2w + ...
             + (data.Y - sig)'*data.X;

    gradz = gradz';
    if nargout > 2
       hessian = - (1/s2w)*eye(n) + data.X'*(diag(sig.*(1-sig))*data.X);
    end
end
