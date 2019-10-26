function demo_radfordnealexample_baselines(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
addpath data/;
outdir= '../diagrams_tables/';

outName = 'binaryclassification';


for dataName = {'Australian' 'German' 'Heart' 'Pima' 'Ripley' 'Caravan'}
%
disp(['Running with ' dataName{1} ' dataset']);
[data.X, data.Y] = loadBinaryClass(dataName{1});
[data.N, data.D] = size(data.X);
data.K = 2;
data.X = [data.X, ones(data.N,1)];
data.titleY = 2*data.Y-1;
s2w = 1;

n=data.D+1
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


% metropolis Hastings
clear mcmc;
mcmc.algorithm = @rwm;
mcmc.inargs{1} = 0.5/n; % initial step size parameter delta
mcmc.inargs{2} = Burn;
mcmc.inargs{3} = T;
mcmc.inargs{4} = adapt;
[tmp, samples, extraOutputs] = mcmc.algorithm(x0, target, mcmc.inargs{:});
% compute statistics
for j=1:size(samples,2)
  summary_mh.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
end
summary_mh.meanW = mean(samples);
summary_mh.varW  = var(samples);
summary_mh.logpxHist = extraOutputs.logpxzHist;
summary_mh.elapsed = extraOutputs.elapsed;
summary_mh.accRate = extraOutputs.accRate;
summary_mh.delta = extraOutputs.delta;
if rep == 1
  summary_mh.samples = samples;
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
   save(['../results/' outName dataName{1} '_baselines_repeat' num2str(rep) '.mat'], 'summary_hmc', 'summary_mh', 'summary_mala');
end

clear summary_hmc summary_mh summary_mala;

end
