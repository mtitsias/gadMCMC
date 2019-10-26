function [out, gradz] = logdensityGaussianDiagonalMixture(x, ww, mm, ss)
% x: input vector
% ww: mixture weights (vector)
% mm: Gaussian means (one row per component)
% ss: standard deviations (one row per component)

aux = bsxfun(@minus, x, mm)./ss;
logp = log(ww)' - 0.5*length(x)*log(2*pi) - sum(log(ss),2) - 0.5*sum(aux.*aux,2);
out = logsumexp(logp, 1);

if nargout > 1
    p_comp = softmax(logp, 1);
    gradz =  -sum(bsxfun(@times, p_comp, aux./ss),1);
end
