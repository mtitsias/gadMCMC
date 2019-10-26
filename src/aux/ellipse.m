function [o1,o2]=ellipse(mu,cmat,varargin)
%ELLIPSE Draws an ellipse
% ellipse(mu,cmat) draws an ellipse centered at mu and with
% covariance matrix cmat.
% h = ellipse(...) plots and returns handle to the ellipse line
% [x,y] = ellipse(...) returns ellipse points (no plot)
% use chiqf(0.9,2)*cmat to get 90% probability region
% example:
% plot(x(:,1),x(:,2),'.'); % plot data points
% hold on; ellipse(xmean, chiqf(0.9,2)*xcov); hold off

% Marko Laine <Marko.Laine@Helsinki.FI>
% $Revision: 1.5 $  $Date: 2006/03/27 13:54:31 $

if size(cmat)~=[2 2]
  error('covmat must be 2x2');
end
if cmat~=cmat'
  error('covmat must be symmetric');
end

t = linspace(0,2*pi)';
R = chol(cmat);
x = mu(1) + R(1,1).*cos(t); 
y = mu(2) + R(1,2).*cos(t) + R(2,2).*sin(t);

if nargout<2
  hh = plot(x,y,varargin{:});
else
  o1 = x;
  o2 = y;
end

if nargout==1
  o1=hh;
end
