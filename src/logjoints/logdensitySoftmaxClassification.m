function [out, gradz] = logdensitySoftmaxClassification(z, data, s2w) 
% z    : 1 x K*(D+1)  latent variable/parameters
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

n = length(z);

Xz = data.X*reshape(z, data.K,data.D+1)';
M = max(Xz, [], 2);
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(sum(sum(z.*z)))/s2w...
       + sum(sum( data.Y.*Xz )) - sum(M) - sum( log( sum(exp( bsxfun(@minus, Xz, M)),2) ) );

if nargout > 1
    % Evaluate gradient of the model
    tmp = (data.Y - softmax(Xz))'*data.X;
    gradz = - z/s2w + tmp(:)'; 
end
