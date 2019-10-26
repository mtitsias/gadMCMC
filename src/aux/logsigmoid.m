function y = logsigmoid(x)

y = zeros(size(x));
idxPos = (x>0);
y(idxPos) = -log(1+exp(-x(idxPos)));
y(~idxPos) = x(~idxPos) - log(1+exp(x(~idxPos)));
