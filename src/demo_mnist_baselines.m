function demo_mnist_baselines(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
addpath data/;
outdir= '../diagrams_tables/';
outName = 'mnist';

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
target.logdensity = @logdensityLogisticRegression;
target.inargs{1} = data;
target.inargs{2} = s2w;

x0 = randn(1,n);

LFs = [5, 10, 20];
% HMC
Burn = 20000;
T = 20000;
adapt = 1;
StoreEvery = 1; 

for i=1:length(LFs)
%
LF = LFs(i);  % leap frog steps
mcmc.algorithm = @hmc;
mcmc.inargs{1} = 0.5/n; % initial step size parameter delta
mcmc.inargs{2} = Burn;
mcmc.inargs{3} = T;
mcmc.inargs{4} = adapt;
mcmc.inargs{5} = LF;
[tmp, samples, extraOutputs] = mcmc.algorithm(x0, target, mcmc.inargs{:});
clear mcmc;
% compute statistics
for j=1:size(samples,2)
summary_hmc{i}.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_hmc{i}.meanW = mean(samples);
summary_hmc{i}.varW  = var(samples);
summary_hmc{i}.LF = LF;
summary_hmc{i}.logpxHist = extraOutputs.logpxzHist;
summary_hmc{i}.elapsed = extraOutputs.elapsed;
summary_hmc{i}.accRate = extraOutputs.accRate;
summary_hmc{i}.delta = extraOutputs.delta;
if rep == 1
  summary_hmc{i}.samples = samples;
end
%
end

% mala
clear mcmc;
mcmc.algorithm = @mala;
mcmc.inargs{1} = 0.5/n; % initial step size parameter delta
mcmc.inargs{2} = Burn;
mcmc.inargs{3} = T;
mcmc.inargs{4} = adapt;
[tmp, samples, extraOutputs] = mcmc.algorithm(x0, target, mcmc.inargs{:});
% compute statistics
for j=1:size(samples,2)
  summary_mala.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_mala.meanW = mean(samples);
summary_mala.varW  = var(samples);
summary_mala.logpxHist = extraOutputs.logpxzHist;
summary_mala.elapsed = extraOutputs.elapsed;
summary_mala.accRate = extraOutputs.accRate;
summary_mala.delta = extraOutputs.delta;
if rep == 1
  summary_mala.samples = samples;
end
if storeRes == 1
   save(['../results/' outName '_baselines_repeat' num2str(rep) '.mat'], 'summary_hmc', 'summary_mala');
end
clear summary_hmc summary_mala;

