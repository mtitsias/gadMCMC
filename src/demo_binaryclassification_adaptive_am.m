function demo_binaryclassification_adaptive_am(rep)

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
  n=data.D+1;
  target.logdensity = @logdensityLogisticRegression;
  target.inargs{1} = data;
  target.inargs{2} = s2w;

  x0 = randn(1,n);

  % MCMC ALGORITHM
  Burn = 20000;
  T = 20000;
  adapt = 1;
  StoreEvery = 1;

  % PROPOSAL DISTRIBUTION
  diagL = 0;
  if diagL == 1
     L = ones(1,n);
  else
     L = eye(n);
     maskL = triu(ones(n))';
  end
  delta = 0.1/sqrt(n);

  [x, samples, extraOutputs] = am(x0, target, delta, L, Burn, T, adapt, StoreEvery);

  % compute statistics
  for j=1:size(samples,2)
   summary_adaptive_am.essW(j) = CalculateESS(samples(:,j), size(samples,1)-1);
  end
  summary_adaptive_am.meanW = mean(samples);
  summary_adaptive_am.varW  = var(samples);
  summary_adaptive_am.logpxHist = extraOutputs.logpxHist;
  summary_adaptive_am.L = extraOutputs.L;
  summary_adaptive_am.elapsed = extraOutputs.elapsed;
  summary_adaptive_am.accRate = extraOutputs.accRate;
  summary_adaptive_am.delta = extraOutputs.delta;
  if rep == 1
    summary_adaptive_am.samples = samples;
  end

  if storeRes == 1
     save(['../results/' outName dataName{1} '_adaptive_am_repeat' num2str(rep) '.mat'], 'summary_adaptive_am');
  end
  clear summary_adaptive_am;
%
end
