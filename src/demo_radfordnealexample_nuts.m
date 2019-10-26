function demo_radfordnealexample_nuts(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
addpath ../../../NUTS-matlab-master/;
addpath ../../../NUTS-matlab-master/Example;
outdir= '../diagrams_tables/';
outName = 'radfordneal_nuts';

n=100;
mu = zeros(n,1);
priorL = diag(0.01:0.01:1);
%target.logdensity = @logdensityGaussian;
target.logdensity = @(x) logdensityGaussian_nuts(x, mu, priorL);
%target.inargs{1} = mu;  % mean vector data
%target.inargs{2} = priorL;   % Cholesky decomposition of the covariacne matrix

x0 = randn(n,1);

xMin = -3; xMax = 3;      % For plotting
yMin = -5; yMax = 1;

% MCMC ALGORITHM
Burn = 500;
T = 20000;
tic; 
[samples, logp_samples] = NUTS_wrapper(target.logdensity, x0, Burn, T);
timetaken=toc; 

samples = samples';
% compute statistics
for j=1:size(samples,2)
  summary_nuts.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_nuts.meanW = mean(samples);
summary_nuts.varW  = var(samples);
summary_nuts.logpxHist = logp_samples;
summary_nuts.elapsed = timetaken;
if rep == 1
  summary_nuts.samples = samples;
end

if storeRes == 1
   save(['../results/' outName '_repeat' num2str(rep) '.mat'], 'summary_nuts');
end
