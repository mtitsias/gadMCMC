clear; 
randn('seed',1);
rand('seed',1);

addpath mcmc/;
addpath aux/;
addpath logjoints/;
outdir= '../diagrams_tables/';
outName = 'Gaussian2D';
FontSz = 18;

mu = [0 0];
Sigma = [1 0.99; 0.99 1];
priorL = chol(Sigma)';
a = 1;
b = 1;
target.logdensity = @logdensityGaussian; %@logdensityBanana2D;
target.inargs{1} = mu;  % mean vector data
target.inargs{2} = priorL;   % Cholesky decomposition of the covariacne matrix

n = 2;
x0 = zeros(1,n);

xMin = -3; xMax = 3;      % For plotting
yMin = -3; yMax = 3;


% MCMC ALGORITHM
Burn = 20000;
T = 1000;
StoreEvery = 1;
adapt = 1;
Optimizer = 1;

% PROPOSAL DISTRIBUTION
diagL = 0;
if diagL == 1 
   L = (0.1/sqrt(n))*ones(1,n);
else
   L = (0.1/sqrt(n))*eye(n);
end
beta = 1;
rho_L = 0.0005;
[x, samples, extraOutputs] = gad_rwm(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer);

L = extraOutputs.L;

figure;
hold on;
x = linspace(xMin,xMax);
y = linspace(yMin,yMax);
[X,Y] = meshgrid(x,y);
for i=1:length(x)
for j=1:length(y)
  Z(i,j) = target.logdensity([x(i), y(j)], target.inargs{:});
  Zproposal(i,j) = logdensityGaussian([x(i), y(j)], [0 0], L);
end
end
[cs, h] = contour(X,Y,exp(Z)', 2, 'g-', 'Linewidth',2);
contour(X,Y,exp(Zproposal)', 2, 'b-', 'Linewidth',2);
%set(h,'linestyle',':');
hold on;
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_adaptive_randomwalk2'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

