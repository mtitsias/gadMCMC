function y = logsumexp(x, dim)

valMax = max(x, [], dim);
aux = bsxfun(@minus, x, valMax);
y = bsxfun(@plus, valMax, log(sum(exp(aux), dim)));
