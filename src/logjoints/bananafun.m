function y=bananafun(x,ab,inverse)
%BANANAFUN banana shaped function
if nargin<3; inverse=0;end
a = ab(1); b = ab(2); % parameters
y = x;
if inverse
  y(:,1) = x(:,1)./a;
  y(:,2) = x(:,2).*a + a.*b.*(x(:,1).^2 + a^2);
else
  y(:,1) = a.*x(:,1);
  y(:,2) = x(:,2)./a - b.*(y(:,1).^2 + a^2);
end
