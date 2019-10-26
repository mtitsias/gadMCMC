function [z, samples, extraOutputs] = mala(z0, log_pxz, delta, Burn, T, adapt, StoreEvery)
% 

if nargin < 7
  StoreEvery = 1;
end

z = z0; 
n = length(z0);  

acceptHist = zeros(1, Burn+T); 
logpxzHist = zeros(1, Burn+T);
% collected samples 
num_stored = floor(T/StoreEvery);
samples = zeros(num_stored,n);

if (Burn+T) == 0
    extraOutputs.delta = delta; 
    extraOutputs.accRate = 0;
    return;
end

[logpxz, gradz] = log_pxz.logdensity(z0, log_pxz.inargs{:});

epsilon = 0.01;
opt = 0.54;
cnt = 0;
tic;  
for i=1:(Burn + T)
%
   znew = z + (delta/2)*gradz + sqrt(delta)*randn(1,n); 
   
   [logpxznew, gradznew] = log_pxz.logdensity(znew, log_pxz.inargs{:});
   
   tmp = znew - z - (delta/2)*gradz;
   proposal_new_given_old = - (0.5/delta)*(tmp*tmp'); % we ignore constants
   
   tmp = z - znew - (delta/2)*gradznew;
   proposal_old_given_new = - (0.5/delta)*(tmp*tmp'); % we ignore constants
    
   %corrFactor = proposal_old_given_new - proposal_new_given_old;  
   accept = MHstep(logpxznew, logpxz, proposal_new_given_old, proposal_old_given_new); 
   
   acceptHist(i) = accept;   
      
   if accept == 1
      z = znew;
      gradz = gradznew; 
      logpxz = logpxznew; 
   end
   
   % Adapt step size delta only during burn-in. After that  
   % collect samples  
   if (i <= Burn) && (adapt == 1) 
      delta = delta + epsilon*((accept - opt)/opt)*delta;
   elseif (mod(i,StoreEvery) == 0)
      cnt = cnt + 1;
      samples(cnt,:) = z;
   end
   logpxzHist(i) = logpxz;
%
end
timetaken=toc; 

extraOutputs.logpxzHist = logpxzHist; 
extraOutputs.acceptHist = acceptHist;
extraOutputs.delta = delta;
extraOutputs.elapsed = timetaken; 
extraOutputs.accRate = mean(acceptHist(Burn+1:end));
