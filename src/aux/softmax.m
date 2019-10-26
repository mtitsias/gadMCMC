function S = softmax(Y,dim) 
%function S = softmax(Y,dim) 
%
%

if(nargin==1)
    dim = 2;
end

Y = bsxfun(@minus, Y, max(Y,[],dim)); 
Y = exp(Y); 
S = bsxfun(@rdivide, Y, sum(Y,dim)); 
