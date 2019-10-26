function [out, gradz] = logdensityGaussianMixture(x, ww, mm, L)
% x: input vector
% ww: mixture weights (vector)
% mm: Gaussian means (one row per component)
% L: Cholesky decomposition of the covariance matrices (3D tensor)

K = length(ww);
C = -0.5*length(x)*log(2*pi);
logp = zeros(1, K);
aux = zeros(length(x), K);
for kk=1:K
    aux(:,kk) = L(:,:,kk)\((x - mm(kk,:))');
    logp(kk) =  log(ww(kk)) + C - sum(log(diag(L(:,:,kk)))) - 0.5*(aux(:,kk)'*aux(:,kk));
end
out = logsumexp(logp, 2);

if nargout > 1
    gradz = zeros(size(x));
    p_comp = softmax(logp, 2);
    for kk=1:K
        gradz = gradz - p_comp(kk)*(L(:,:,kk)'\aux(:,kk))';
    end
end
