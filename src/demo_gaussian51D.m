function demo_gaussian51D(rep, Optimizer)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath data/;
addpath logjoints/;
outdir= '../diagrams_tables/';
outName = ['gpregression_adaptive_mala_exact_optim' num2str(Optimizer)];

% Fix seeds
%randn('seed', 1e5);
%rand('seed', 1e5);
%sigma2 = 0.05;
step = 0.08; 
X = (0:step:4)';
logtheta = 2*mean(log((max(X) - min(X))*0.1));
exp(logtheta)
[n D] = size(X);
X = X./exp(logtheta/2); 
jitter = 0.01;
Knn = -2*X*X' + repmat(sum(X.*X,2)',n,1) + repmat(sum(X.*X,2),1,n);
Knn = exp(-0.5*Knn') + jitter*eye(n); 
priorL = chol(Knn)';
mu = zeros(1,n);
target.logdensity = @logdensityGaussian;
target.inargs{1} = mu;     % mean prior vector
target.inargs{2} = priorL; % Cholesky decomposition of the covariance matrix
target.inargs{3} = - priorL'\(priorL\eye(n));

% initial mcmc state
x0 = randn(1,n);

% MCMC ALGORITHM
Burn = 200000;
T = 20000;
adapt = 1;
StoreEvery = 1;

% PROPOSAL DISTRIBUTION
diagL = 0;
if diagL == 1 
   L = (0.1/sqrt(n))*ones(1,n);
else
   L = (0.1/sqrt(n))*eye(n);
end
beta = 1;
rho_L = 0.001;
[x, samples, extraOutputs] = gad_mala_fast(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer);

% compute statistics
for j=1:size(samples,2)
  summary_adaptive_mala_exact.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_adaptive_mala_exact.meanW = mean(samples);
summary_adaptive_mala_exact.varW  = var(samples);
summary_adaptive_mala_exact.logpxHist = extraOutputs.logpxHist;
summary_adaptive_mala_exact.lowerboundHist = extraOutputs.lowerboundHist;
summary_adaptive_mala_exact.L = extraOutputs.L;
summary_adaptive_mala_exact.elapsed = extraOutputs.elapsed;
summary_adaptive_mala_exact.accRate = extraOutputs.accRate;
summary_adaptive_mala_exact.beta = extraOutputs.beta;
if rep == 1
  summary_adaptive_mala_exact.samples = samples;
end

FontSz = 18;

figure;
imagesc(Knn); 
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_adaptive_mala_Knn'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

figure;
imagesc(summary_adaptive_mala_exact.L*summary_adaptive_mala_exact.L');
set(gca,'fontsize',FontSz);
box on;
name = [outdir outName '_adaptive_mala_LL'];
print('-depsc2', '-r300', name);
cmd = sprintf('epstopdf %s', [name '.eps']);
system(cmd);

if storeRes == 1
    save(['../results/' outName '_repeat' num2str(rep) '.mat'], 'summary_adaptive_mala_exact');
end

