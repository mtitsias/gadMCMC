function [z, samples, extraOutputs] = rwm(z0, log_pxz, delta, Burn, T, adapt)
% 

z = z0; 
n = length(z0);  

acceptHist = zeros(1, Burn+T); 
logpxzHist = zeros(1, Burn+T);
% collected samples 
samples = zeros(T, n); 

if (Burn+T) == 0
    extraOutputs.delta = delta; 
    extraOutputs.accRate = 0;
    return;
end

logpxz = log_pxz.logdensity(z0, log_pxz.inargs{:});
epsilon = 0.01;
opt = 0.25;
cnt = 0; 
tic; 
for i=1:(Burn + T)
%
   znew = z + sqrt(delta)*randn(1,n); 
   
   logpxznew = log_pxz.logdensity(znew, log_pxz.inargs{:});
       
   accept = MHstep(logpxznew, logpxz, 0, 0); 
   
   acceptHist(i) = accept;   
      
   if accept == 1
      z = znew;
      logpxz = logpxznew; 
   end
   
   % Adapt step size delta only during burn-in. After that  
   % collect samples  
   if (i <= Burn) && (adapt == 1) 
      delta = delta + epsilon*((accept - opt)/opt)*delta;
   else
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
