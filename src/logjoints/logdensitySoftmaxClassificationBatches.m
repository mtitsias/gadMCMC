function [out, gradz] = logdensitySoftmaxClassificationBatches(z, data, s2w) 
% z    : 1 x K*(D+1)  latent variable/parameters
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

n = length(z);
sviFactor = data.N / size(data.batch.Xminibatch,1);

Xz = data.batch.Xminibatch*reshape(z, data.K,data.D+1)';
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(sum(sum(z.*z)))/s2w...
       + sviFactor * ( sum(sum( data.Y(data.batch.block,:).*Xz )) - sum(logsumexp(Xz, 2)) );

if nargout > 1
    % Evaluate gradient of the model
    tmp = (data.Y(data.batch.block,:) - softmax(Xz))'*data.batch.Xminibatch;
    gradz = - z/s2w + sviFactor*tmp(:)'; 
end
