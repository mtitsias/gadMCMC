function demo_binaryclassification_adaptive_randomwalk(rep, Optimizer)

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
     L = (0.1/sqrt(n))*ones(1,n);
  else
     L = (0.1/sqrt(n))*eye(n);
     maskL = triu(ones(n))';
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
     save(['../results/' outName dataName{1} '_adaptive_randomwalk_repeat' num2str(rep) '.mat'], 'summary_adaptive_randomwalk');
  end
  clear summary_adaptive_randomwalk x samples target L extraOutputs data;
%
end
