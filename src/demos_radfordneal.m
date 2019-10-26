% - Radford Neals's runs and creating plots and results'
%
randn('seed', 0);
rand('seed', 0);
% how many times to repeat the experiments 
Repeats = 10; 
Optimizer = 1; 
for r=1:Repeats
  demo_radfordnealexample_adaptive_randomwalk(r, Optimizer);
  demo_radfordnealexample_adaptive_mala_fast(r, Optimizer);
  demo_radfordnealexample_adaptive_mala_exact(r, Optimizer);
  demo_radfordnealexample_adaptive_am(r);
  demo_radfordnealexample_baselines(r);
end

outdir = '../diagrams_tables/';
fontsz = 26;
addpath ../results/;

outName = 'radfordneal';

LF = [5,10,20];

for rep=1:Repeats
%
  load(['radfordneal_adaptive_randomwalk_optim' num2str(Optimizer) '_repeat' num2str(rep) '.mat']);

  ESSmin_adaptive_randomwalk(rep) = min(summary_adaptive_randomwalk.essW);
  ESSmedian_adaptive_randomwalk(rep) = median(summary_adaptive_randomwalk.essW);
  ESSmax_adaptive_randomwalk(rep) = max(summary_adaptive_randomwalk.essW);

  TrainTime_adaptive_randomwalk(rep) = summary_adaptive_randomwalk.elapsed;
  delta_adaptive_randomwalk(rep) = min(min(summary_adaptive_randomwalk.L));
  beta_adaptive_randomwalk(rep) = summary_adaptive_randomwalk.beta;
  accRate_adaptive_randomwalk(rep) = summary_adaptive_randomwalk.accRate;

  load(['radfordneal_adaptive_mala_optim' num2str(Optimizer) '_repeat' num2str(rep) '.mat']);

  ESSmin_adaptive_mala(rep) = min(summary_adaptive_mala.essW);
  ESSmedian_adaptive_mala(rep) = median(summary_adaptive_mala.essW);
  ESSmax_adaptive_mala(rep) = max(summary_adaptive_mala.essW);

  TrainTime_adaptive_mala(rep) = summary_adaptive_mala.elapsed;
  delta_adaptive_mala(rep) = min(min(summary_adaptive_mala.L));
  beta_adaptive_mala(rep) = summary_adaptive_mala.beta;
  accRate_adaptive_mala(rep) = summary_adaptive_mala.accRate;

  load(['radfordneal_adaptive_mala_exact_optim' num2str(Optimizer) '_repeat' num2str(rep) '.mat']);

  ESSmin_adaptive_mala_exact(rep) = min(summary_adaptive_mala_exact.essW);
  ESSmedian_adaptive_mala_exact(rep) = median(summary_adaptive_mala_exact.essW);
  ESSmax_adaptive_mala_exact(rep) = max(summary_adaptive_mala_exact.essW);

  TrainTime_adaptive_mala_exact(rep) = summary_adaptive_mala_exact.elapsed;
  delta_adaptive_mala_exact(rep) = min(min(summary_adaptive_mala_exact.L));
  beta_adaptive_mala_exact(rep) = summary_adaptive_mala_exact.beta;
  accRate_adaptive_mala_exact(rep) = summary_adaptive_mala_exact.accRate;

  
  load(['radfordneal_adaptive_am_repeat' num2str(rep) '.mat']);

  ESSmin_am(rep) = min(summary_adaptive_am.essW);
  ESSmedian_am(rep) = median(summary_adaptive_am.essW);
  ESSmax_am(rep) = max(summary_adaptive_am.essW);
  TrainTime_am(rep) = summary_adaptive_am.elapsed;
  accRate_am(rep) = summary_adaptive_am.accRate;
  
  load(['radfordneal_nuts_repeat' num2str(rep) '.mat']);

  ESSmin_nuts(rep) = min(summary_nuts.essW);
  ESSmedian_nuts(rep) = median(summary_nuts.essW);
  ESSmax_nuts(rep) = max(summary_nuts.essW);
  TrainTime_nuts(rep) = summary_nuts.elapsed;
  
  load(['radfordneal_baselines_repeat' num2str(rep) '.mat']);

  ESSmin_mh(rep) = min(summary_mh.essW);
  ESSmedian_mh(rep) = median(summary_mh.essW);
  ESSmax_mh(rep) = max(summary_mh.essW);
  TrainTime_mh(rep) = summary_mh.elapsed;
  delta_mh(rep) = summary_mh.delta;
  accRate_mh(rep) = summary_mh.accRate;

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

  n3 = 20000;
  if 1
  ylim1 = -100;
  ylim2 = -10;
  gap = 4;
  if rep == 1
    % plot the evolution of the log target
    figure;
    nn = length(summary_adaptive_randomwalk.logpxHist);
    plot(1:gap:nn, summary_adaptive_randomwalk.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_adaptive_randomwalk.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('gadRWM');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_randomwalk'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(diag(summary_adaptive_randomwalk.L),'k', 'linewidth',2);
    %ylabel('Diagonal of L','fontsize',fontsz);
    title('gadRWM');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_randomwalk_L'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_adaptive_mala.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_adaptive_mala.logpxHist) ylim1 ylim2]); 
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
    plot(diag(summary_adaptive_mala.L),'k', 'linewidth',2);
    %ylabel('Diagonal of L','fontsize',fontsz);
    title('gadMALAf');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_mala_L'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);


    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_adaptive_mala_exact.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_adaptive_mala_exact.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('gadMALAe');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_mala_exact'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(diag(summary_adaptive_mala_exact.L),'k', 'linewidth',2);
    %ylabel('Diagonal of L','fontsize',fontsz);
    title('gadMALAe');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_mala_exact_L'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_mh.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_mh.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('RWM');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_mh'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_mala.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_mala.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title('MALA');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_mala'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

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
    plot(1:2:nn2, summary_nuts.logpxHist(1:2:end),'k.', 'linewidth',1);
    axis([1 length(summary_nuts.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title(['NUTS']);
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_nuts'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
    
    % plot the evolution of the log target
    figure;
    plot(1:gap:nn, summary_adaptive_am.logpxHist(1:gap:end),'k.', 'linewidth',1);
    axis([1 length(summary_adaptive_am.logpxHist) ylim1 ylim2]); 
    ylabel('Log-target','fontsize',fontsz);
    title(['AM']);
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_am'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);

    figure;
    plot(summary_adaptive_am.L,'k', 'linewidth',2);
    %ylabel('Diagonal L','fontsize',fontsz);
    title('AM');
    set(gca,'fontsize',fontsz);
    box on;
    name = [outdir outName '_adaptive_am_L'];
    print('-depsc2', '-r300', name);
    cmd = sprintf('epstopdf %s', [name '.eps']);
    system(cmd);
    
    Dims = [1,100];
    step = 0.01; 
    for d = Dims
      figure;
      plot(1:2:n3, summary_adaptive_randomwalk.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_adaptive_randomwalk.samples(:,d)) -4*d*step 4*d*step]);
      title('gadRWM');
      %ylabel('Log-joint','fontsize',fontsz);
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_adaptive_randomwalk_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);

      figure;
      plot(1:2:n3, summary_adaptive_mala.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_adaptive_mala.samples(:,d)) -4*d*step 4*d*step]);
      title('gadMALAf');
      %ylabel('Log-joint','fontsize',fontsz);
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_adaptive_mala_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);

      figure;
      plot(1:2:n3, summary_adaptive_mala_exact.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_adaptive_mala_exact.samples(:,d)) -4*d*step 4*d*step]);
      title('gadMALAe');
      %ylabel('Log-joint','fontsize',fontsz);
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_adaptive_mala_exact_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);

      figure;
      plot(1:2:n3, summary_mh.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_mh.samples(:,d)) -4*d*step 4*d*step]);
      title('RWM');
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_mh_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);

      figure;
      plot(1:2:n3, summary_mala.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_mala.samples(:,d)) -4*d*step 4*d*step]);
      title('MALA');
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_mala_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);

      for gg=1:length(summary_hmc)
      figure;
      plot(1:2:n3, summary_hmc{gg}.samples(1:2:end,d),'k.', 'linewidth',1);
      axis([1 length(summary_hmc{gg}.samples(:,d)) -4*d*step 4*d*step]);
      title(['HMC-' num2str(LF(gg))]);
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_hmc_dim_' num2str(d) '_LF_' num2str(gg)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);
      end
      
      figure;
      plot(1:2:n3, summary_nuts.samples(1:2:n3,d),'k.', 'linewidth',1);
      axis([1 length(summary_nuts.samples(:,d)) -4*d*step 4*d*step]);
      title('NUTS');
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_nuts_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);
      
      figure;
      plot(1:2:n3, summary_adaptive_am.samples(1:2:n3,d),'k.', 'linewidth',1);
      axis([1 length(summary_adaptive_am.samples(:,d)) -4*d*step 4*d*step]);
      title('AM');
      %ylabel('Log-joint','fontsize',fontsz);
      set(gca,'fontsize',fontsz);
      box on;
      name = [outdir outName '_adaptive_am_dim_' num2str(d)];
      print('-depsc2', '-r300', name);
      cmd = sprintf('epstopdf %s', [name '.eps']);
      system(cmd);
      
    end

  end
  end
%
end

fid = fopen([outdir outName '_table.txt'],'w');
fprintf(fid,'Method &  Time(s) & Accept Rate &  ESS (Min, Med, Max)  & Min ESS/s (1 st.d.) \\\\ \n');
fprintf(fid,'\\midrule\n');
fprintf(fid,'gadMALAf  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_adaptive_mala), mean(accRate_adaptive_mala), mean(ESSmin_adaptive_mala), mean(ESSmedian_adaptive_mala), mean(ESSmax_adaptive_mala), mean(ESSmin_adaptive_mala./TrainTime_adaptive_mala), std(ESSmin_adaptive_mala./TrainTime_adaptive_mala));
fprintf(fid,'gadMALAe  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_adaptive_mala_exact), mean(accRate_adaptive_mala_exact), mean(ESSmin_adaptive_mala_exact), mean(ESSmedian_adaptive_mala_exact), mean(ESSmax_adaptive_mala_exact), mean(ESSmin_adaptive_mala_exact./TrainTime_adaptive_mala_exact), std(ESSmin_adaptive_mala_exact./TrainTime_adaptive_mala_exact));
fprintf(fid,'gadRWM  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_adaptive_randomwalk), mean(accRate_adaptive_randomwalk), mean(ESSmin_adaptive_randomwalk), mean(ESSmedian_adaptive_randomwalk), mean(ESSmax_adaptive_randomwalk), mean(ESSmin_adaptive_randomwalk./TrainTime_adaptive_randomwalk), std(ESSmin_adaptive_randomwalk./TrainTime_adaptive_randomwalk));
fprintf(fid,'AM  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_am), mean(accRate_am), mean(ESSmin_am), mean(ESSmedian_am), mean(ESSmax_am), mean(ESSmin_am./TrainTime_am), std(ESSmin_am./TrainTime_am));
fprintf(fid,'RWM  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_mh), mean(accRate_mh), mean(ESSmin_mh), mean(ESSmedian_mh), mean(ESSmax_mh), mean(ESSmin_mh./TrainTime_mh), std(ESSmin_mh./TrainTime_mh));
fprintf(fid,'MALA  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_mala), mean(accRate_mala), mean(ESSmin_mala), mean(ESSmedian_mala), mean(ESSmax_mala), mean(ESSmin_mala./TrainTime_mala), std(ESSmin_mala./TrainTime_mala)); 
for i = 1:length(TrainTime_hmc)
fprintf(fid,'%s HMC  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n','%', ...
        mean(TrainTime_hmc{i}), mean(accRate_hmc{i}), mean(ESSmin_hmc{i}), mean(ESSmedian_hmc{i}), mean(ESSmax_hmc{i}), mean(ESSmin_hmc{i}./TrainTime_hmc{i}), std(ESSmin_hmc{i}./TrainTime_hmc{i}));
  if i == 1
     bi = 1;
     bestval = mean(ESSmin_hmc{i}./TrainTime_hmc{i});
  elseif bestval < mean(ESSmin_hmc{i}./TrainTime_hmc{i});
     bi = i; 
     bestval = mean(ESSmin_hmc{i}./TrainTime_hmc{i});
  end
end
% report the best performing HMC method 
fprintf(fid,'HMC-%d  &   %1.1f  &  %1.3f  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n',LF(bi),...
        mean(TrainTime_hmc{bi}), mean(accRate_hmc{bi}), mean(ESSmin_hmc{bi}), mean(ESSmedian_hmc{bi}), mean(ESSmax_hmc{bi}), mean(ESSmin_hmc{bi}./TrainTime_hmc{bi}), std(ESSmin_hmc{bi}./TrainTime_hmc{bi}));
fprintf(fid,'NUTS  &   %1.1f  &  %s  &  (%1.1f, %1.1f, %1.1f)  &  %1.2f (%1.2f)\\\\ \n', ...
        mean(TrainTime_nuts), '>0.7', mean(ESSmin_nuts), mean(ESSmedian_nuts), mean(ESSmax_nuts), mean(ESSmin_nuts./TrainTime_nuts), std(ESSmin_nuts./TrainTime_nuts));
fclose(fid);
