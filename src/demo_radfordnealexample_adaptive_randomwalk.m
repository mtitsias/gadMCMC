function demo_radfordnealexample_adaptive_randomwalk(rep, Optimizer)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
outdir= '../diagrams_tables/';

outName = ['radfordneal_adaptive_randomwalk_optim' num2str(Optimizer)];

n=100;
mu = zeros(1,n);
priorL = diag(0.01:0.01:1);
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
diagL = 0;
if diagL == 1
   L = (0.1/sqrt(n))*ones(1,n);
   extraArgs.Sigma = (diag(priorL)').^2;
else
   L = (0.1/sqrt(n))*eye(n);
   maskL = triu(ones(n))';
   extraArgs.Sigma = priorL*priorL';
end
beta = 1;

% baseline learning rate for RMSprop
rho_L = 0.00005;
[x, samples, extraOutputs] = gad_rwm(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer);

% compute statistics
for j=1:size(samples,2)
  summary_adaptive_randomwalk.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_adaptive_randomwalk.meanW = mean(samples);
summary_adaptive_randomwalk.varW  = var(samples);
summary_adaptive_randomwalk.logpxHist = extraOutputs.logpxHist;
summary_adaptive_randomwalk.lowerboundHist = extraOutputs.lowerboundHist;
summary_adaptive_randomwalk.L = extraOutputs.L;
summary_adaptive_randomwalk.elapsed = extraOutputs.elapsed;
summary_adaptive_randomwalk.accRate = extraOutputs.accRate;
summary_adaptive_randomwalk.beta = extraOutputs.beta;

if rep == 1
  summary_adaptive_randomwalk.samples = samples;
end

if storeRes == 1
   save(['../results/' outName '_repeat' num2str(rep) '.mat'], 'summary_adaptive_randomwalk');
end


if 0
% plot the smoothed new divergence
smoothedbound = smoothedAverage(lowerboundHist, 200);
figure;
plot(smoothedbound,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_bound'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

% plot the evolution of the log target
smoothedbound = smoothedAverage(lowerboundHist, 200);
figure;
plot(logpxHist,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_logdensity'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

figure;
plot(L,'r', 'linewidth',2);
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_L'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);
end
