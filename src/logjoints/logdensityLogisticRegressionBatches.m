function [out, gradz] = logdensityLogisticRegressionBatches(z, data, s2w) 
% z    : (1 x n) latent variable
% data : data struct (it contains 'data.N', 'data.D', 'data.X', 'data.Y')
% s2w  : prior variance on the regression coefficients

n = length(z);
sviFactor = data.N / size(data.batch.Xminibatch,1);
Xz = data.batch.Xminibatch*z';
out = -0.5*n*log(2*pi) - 0.5*n*log(s2w) - 0.5*(z*z')/s2w ...
      + sviFactor * sum( logsigmoid(data.titleY(data.batch.block).*Xz) );
if nargout > 1
    % Evaluate gradient of the model
    gradz  = - z/s2w + ...
             + sviFactor * (data.Y(data.batch.block) - sigmoid(Xz))'*data.batch.Xminibatch;
end
