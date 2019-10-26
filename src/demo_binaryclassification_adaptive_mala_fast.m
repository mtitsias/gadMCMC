function demo_binaryclassification_adaptive_mala_fast(rep, Optimizer)

storeRes = 1;

addpath mcmc/;
addpath aux/;
addpath data/;
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
  % target distribution
  target.logdensity = @logdensityLogisticRegression;
  target.inargs{1} = data;
  target.inargs{2} = s2w;

  % initial mcmc state
  x0 = randn(1,n);

  % MCMC ALGORITHM
  Burn = 20000;
  T = 20000;
  adapt = 1;
  StoreEvery = 1;

  % PROPOSAL DISTRIBUTION
  L = (0.1/sqrt(n))*eye(n);
  beta = 1;

  % baseline learning for RMSprop
  rho_L = 0.00015;
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

  min(summary_adaptive_mala.essW)
  if storeRes == 1
    save(['../results/' outName dataName{1} '_adaptive_mala_repeat' num2str(rep) '.mat'], 'summary_adaptive_mala');
  end
  clear summary_adaptive_mala x samples target L extraOutputs data;
end
