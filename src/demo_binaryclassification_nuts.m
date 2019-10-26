function demo_binaryclassification_nuts(rep)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath logjoints/;
addpath data/;
addpath ../../../NUTS-matlab-master/;
addpath ../../../NUTS-matlab-master/Example;
outdir= '../diagrams_tables/';
outName = 'binaryclassification';

for dataName = {'German'}% {'Australian' 'German' 'Heart' 'Pima' 'Ripley' 'Caravan'}
%
  disp(['Running with ' dataName{1} ' dataset']);
  [data.X, data.Y] = loadBinaryClass(dataName{1});
  [data.N, data.D] = size(data.X);
  data.K = 2;
  data.X = [data.X, ones(data.N,1)];
  data.titleY = 2*data.Y-1;
  s2w = 1;
  n=data.D+1;
  %target.logdensity = @logdensityLogisticRegression;
  target.logdensity = @(x) logdensityLogisticRegression_nuts(x, data, s2w);
  %target.inargs{1} = data;
  %target.inargs{2} = s2w;

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
     save(['../results/' outName dataName{1} '_nuts_repeat' num2str(rep) '.mat'], 'summary_nuts');
  end
  clear summary_nuts;
%
end
