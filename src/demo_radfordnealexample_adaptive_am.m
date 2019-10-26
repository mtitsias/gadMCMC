function demo_radfordnealexample_adaptive_am(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
outdir= '../diagrams_tables/';
outName = 'radfordneal_adaptive_am';

n=100;
mu = zeros(1,n);
priorL = diag(0.01:0.01:1);
a = 1;
b = 1;
target.logdensity = @logdensityGaussian;
target.inargs{1} = mu;  % mean vector data
target.inargs{2} = priorL;   % Cholesky decomposition of the covariacne matrix

x0 = randn(1,n);

xMin = -3; xMax = 3;      % For plotting
yMin = -5; yMax = 1;

% MCMC ALGORITHM
Burn = 20000;
T = 20000;
adapt = 1;
StoreEvery = 1;

% PROPOSAL DISTRIBUTION
diagL = 1;
if diagL == 1 
   L = ones(1,n);
else
   L = eye(n);
   maskL = triu(ones(n))';
end
delta = (0.1/sqrt(n));

[x, samples, extraOutputs] = am(x0, target, delta, L, Burn, T, adapt, StoreEvery);

% compute statistics
for j=1:size(samples,2)
  summary_adaptive_am.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_adaptive_am.meanW = mean(samples);
summary_adaptive_am.varW  = var(samples);
summary_adaptive_am.logpxHist = extraOutputs.logpxHist;
summary_adaptive_am.L = extraOutputs.L;
summary_adaptive_am.delta = extraOutputs.delta;
summary_adaptive_am.elapsed = extraOutputs.elapsed;
summary_adaptive_am.accRate = extraOutputs.accRate;
if rep == 1
  summary_adaptive_am.samples = samples;
end

if storeRes == 1
   save(['../results/' outName '_repeat' num2str(rep) '.mat'], 'summary_adaptive_am');
end


if 0
FontSz = 26;
logpxHist =  extraOutputs.logpxHist;
figure;
plot(logpxHist,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_logdensity'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

figure;
L = extraOutputs.L;
plot(L,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_L'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

end
