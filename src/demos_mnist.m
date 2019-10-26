% - mnist runs and creating plots and results'
%
randn('seed', 0);
rand('seed', 0);
% how many times to repeat the experiments 
Repeats = 5; 
Optimizer = 1; 
for r=1:Repeats
 demo_mnist_adaptive_mala_fast(r, Optimizer);
 demo_mnist_baselines(r);
end

outdir = '../diagrams_tables/';
fontsz = 26;
addpath ../results/;

outName = 'mnist';
LF = [5,10,20];

for rep=1:Repeats
%
  load([outName '_adaptive_mala_repeat' num2str(rep) '.mat']);

  ESSmin_adaptive_mala(rep) = min(summary_adaptive_mala.essW);
  ESSmedian_adaptive_mala(rep) = median(summary_adaptive_mala.essW);
  ESSmax_adaptive_mala(rep) = max(summary_adaptive_mala.essW);

  TrainTime_adaptive_mala(rep) = summary_adaptive_mala.elapsed;
  delta_adaptive_mala(rep) = min(min(summary_adaptive_mala.L));
  beta_adaptive_mala(rep) = summary_adaptive_mala.beta;
  accRate_adaptive_mala(rep) = summary_adaptive_mala.accRate;

  if rep <= 3
    load([outName '_nuts_repeat' num2str(rep) '.mat']);
  else
    rr = 2;  
    load([outName '_nuts_repeat' num2str(rr) '.mat']);
  end    
  summary_nuts
  
  ESSmin_nuts(rep) = min(summary_nuts.essW);
  ESSmedian_nuts(rep) = median(summary_nuts.essW);
  ESSmax_nuts(rep) = max(summary_nuts.essW);
  TrainTime_nuts(rep) = summary_nuts.elapsed;
  
  load([outName '_baselines_repeat' num2str(rep) '.mat']);

  ESSmin_mala(rep) = min(summary_mala.essW);
  ESSmedian_mala(rep) = median(summary_mala.essW);
  ESSmax_mala(rep) = max(summary_mala.essW);
  TrainTime_mala(rep) = summary_mala.elapsed;
  delta_mala(rep) = summary_mala.delta;
  accRate_mala(rep) = summary_mala.accRate;

  for i=1:length(summary_hmc)
   ESSmin_hmc{i}(rep) = min(summary_hmc{i}.essW);
   ESSmedian_hmc{i}(rep) = median(summary_hmc{i}.essW);
   ESSmax_hmc{i}(rep) = max(summary_hmc{i}.essW);
   TrainTime_hmc{i}(rep) = summary_hmc{i}.elapsed;
   delta_hmc{i}(rep) = summary_hmc{i}.delta;
   accRate_hmc{i}(rep) = summary_hmc{i}.accRate;
  end

  gap = 4;
  nn = length(summary_hmc{1}.logpxHist);
  nn
  ylim1 = -3400;
  ylim2 = -3100;
  if rep == 1
    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_adaptive_mala.logpxHist(30001:gap:end),'k.', 'linewidth',1);
    axis([1 nn ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('gadMALAf');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_mala'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_mala.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 nn ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('MALA');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_mala'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
     % plot the evolution of the log target
    for gg=1:length(summary_hmc)
    figure;
    plot(1:gap:nn, summary_hmc{gg}.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_hmc{gg}.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title(['HMC-' num2str(LF(gg))]);
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_hmc_LF_' num2str(gg)];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
    end

    % plot the evolution of the log target
    figure;
    nn2 = length(summary_nuts.logpxHist); 
    plot(1:2:nn2, summary_nuts.logpxHist(1:2:nn2),'k.', 'linewidth',1);
    axis([1 length(summary_nuts.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('NUTS');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_nuts'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
    
    figure; 
    ok = diag(summary_adaptive_mala.L);
    imagesc(reshape(ok(1:784),28,28)');
    colormap('gray');
    %axis off;    
    name = [outdir outName '_ada_L'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
  end
%
end

fid = fopen([outdir outName '_table.txt'],'w');
fprintf(fid,'Method &  Time(s) & Accept Rate &  ESS (Min, Med, Max)  & Min ESS/s (1 st.d.) \\\\ \n');
fprintf(fid,'\\midrule\n');
fprintf(fid,'gadMALAf  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.3f (%1.2f)\\\\ \n', ...
        mean(TrainTime_adaptive_mala), mean(accRate_adaptive_mala), mean(ESSmin_adaptive_mala), mean(ESSmedian_adaptive_mala), mean(ESSmax_adaptive_mala), mean(ESSmin_adaptive_mala./TrainTime_adaptive_mala), std(ESSmin_adaptive_mala./TrainTime_adaptive_mala));
fprintf(fid,'MALA  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.3f (%1.2f)\\\\ \n', ...
        mean(TrainTime_mala), mean(accRate_mala), mean(ESSmin_mala), mean(ESSmedian_mala), mean(ESSmax_mala), mean(ESSmin_mala./TrainTime_mala), std(ESSmin_mala./TrainTime_mala));
for i = 1:length(TrainTime_hmc)
fprintf(fid,'HMC-%d  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.3f (%1.2f)\\\\ \n',LF(i), ...
        mean(TrainTime_hmc{i}), mean(accRate_hmc{i}), mean(ESSmin_hmc{i}), mean(ESSmedian_hmc{i}), mean(ESSmax_hmc{i}), mean(ESSmin_hmc{i}./TrainTime_hmc{i}), std(ESSmin_hmc{i}./TrainTime_hmc{i}));
end
fprintf(fid,'NUTS  &   %1.1f  &  %s  &  (%1.1f, %1.1f, %1.1f)  &  %1.3f (%1.2f)\\\\ \n', ...
        mean(TrainTime_nuts), '>0.7', mean(ESSmin_nuts), mean(ESSmedian_nuts), mean(ESSmax_nuts), mean(ESSmin_nuts./TrainTime_nuts), std(ESSmin_nuts./TrainTime_nuts));
fclose(fid);
