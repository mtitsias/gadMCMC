function demo_radfordnealexamle_adaptive_randomwalk(rep, Optimizer)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath data/;
addpath logjoints/;
outdir= '../diagrams_tables/';
outName = ['radfordneal_adaptive_mala_exact_optim' num2str(Optimizer)];

n=100;
mu = zeros(1,n);
priorL = diag(0.01:0.01:1);
target.logdensity = @logdensityGaussian;
target.inargs{1} = mu;  % mean vector data
target.inargs{2} = priorL;   % Cholesky decomposition of the covariacne matrix
target.inargs{3} = - priorL'\(priorL\eye(n));

% initial mcmc state
x0 = randn(1,n);

% MCMC ALGORITHM
Burn = 20000;
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

% baseline learning for RMSprop
rho_L = 0.00015;
[x, samples, extraOutputs] = gad_mala_exact(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer);

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


if storeRes == 1
    save(['../results/' outName '_repeat' num2str(rep) '.mat'], 'summary_adaptive_mala_exact');
end

