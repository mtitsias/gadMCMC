function demo_mnist_nuts(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath data/;
addpath logjoints/;
addpath ../../../NUTS-matlab-master/;
addpath ../../../NUTS-matlab-master/Example;
outdir= '../diagrams_tables/';
outName = 'mnist_nuts';

% DEFINE THE TARGET DISTRIBUTION
train_data_per_class = 2000;
%[data.X, data.Y] = loadMnist(train_data_per_class);
[data.X, data.Y] = loadMnist();
ind5 = find(data.Y(:,6)==1);
ind6 = find(data.Y(:,7)==1);
data.X = [data.X(ind5, :); data.X(ind6,:)];
[data.N, data.D] = size(data.X);
data.K = 2;
data.X = [data.X, ones(data.N,1)];  
data.Y = ones(data.N, 1);
data.Y(length(ind6)+1:end) = 0;
data.titleY = 2*data.Y-1;
s2w = 1;
n=data.D+1;
% target distribution
target.logdensity = @(x) logdensityLogisticRegression_nuts(x, data, s2w);

% initial mcmc state
x0 = randn(n,1);

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
