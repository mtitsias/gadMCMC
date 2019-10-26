function demo_mnist_adaptive_mala(rep, Optimizer)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath data/;
addpath logjoints/;
outName = 'mnist';

% DEFINE THE TARGET DISTRIBUTION
train_data_per_class = 2000;
%[data.X, data.Y] = loadMnist(train_data_per_class);
[data.X, data.Y] = loadMnist();
ind5 = find(data.Y(:,6)==1);
ind6 = find(data.Y(:,7)==1);
data.X = [data.X(ind5, :); data.X(ind6,:)];
[data.N, data.D] = size(data.X);
data.N
data.K = 2;
data.X = [data.X, ones(data.N,1)];  
data.Y = ones(data.N, 1);
data.Y(length(ind6)+1:end) = 0;
data.titleY = 2*data.Y-1;
s2w = 1;
n=data.D+1;
% target distribution
target.logdensity = @logdensityLogisticRegression;
target.inargs{1} = data;
target.inargs{2} = s2w;

% initial mcmc state
x0 = randn(1,n);

% MCMC ALGORITHM
Burn = 50000;
T = 20000;
adapt = 1;
StoreEvery = 1;
Optimizer = 1;

% PROPOSAL DISTRIBUTION
diagL = 0;
if diagL == 1
  L = (0.1/sqrt(n))*ones(1,n);
else
  L = (0.1/sqrt(n))*eye(n);
end
beta = 1;
rho_L = 0.00001;
[x, samples, extraOutputs] = gad_mala_fast(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer);

% compute statistics
for j=1:size(samples,2)
  summary_adaptive_mala.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_adaptive_mala.meanW = mean(samples);
summary_adaptive_mala.varW  = var(samples);
summary_adaptive_mala.logpxHist = extraOutputs.logpxHist;
summary_adaptive_mala.lowerboundHist = extraOutputs.lowerboundHist;
summary_adaptive_mala.L = extraOutputs.L;
summary_adaptive_mala.elapsed = extraOutputs.elapsed;
summary_adaptive_mala.accRate = extraOutputs.accRate;
summary_adaptive_mala.beta = extraOutputs.beta;
if rep == 1
  summary_adaptive_mala.samples = samples;
end

if storeRes == 1
  save(['../results/' outName '_adaptive_mala_repeat' num2str(rep) '.mat'], 'summary_adaptive_mala');
end
  
  
