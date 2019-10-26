function [x, samples, extraOutputs] = am(x0, target, delta, L, Burn, T, adapt, StoreEvery, extraArgs)
% 

% if extraArgs are given then you compute the 
% inhomogeneity factor b in iteration
flagExtra = 0;  
if nargin == 9
   trueSigma = extraArgs.Sigma;
   flagExtra = 1;
   bfactor = zeros(1, Burn+T);
end 


if nargin < 8
  StoreEvery = 1;
end

x = x0;
n = length(x0);

diagL = 0;
if min(size(L))==1
  diagL = 1;
else
  maskL = triu(ones(n))';
end

acceptHist = zeros(1, Burn+T);
logpxHist = zeros(1, Burn+T);
% collected samples 
num_stored = floor(T/StoreEvery);
samples = zeros(num_stored,n);

if (Burn+T) == 0
    extraOutputs.L = L;
    extraOutputs.accRate = 0;
    return;
end

% LEANRING RATE
rho_L = 0.0005;
T0 = round(Burn/4);  
rho_mu = 0.01;
rho_delta = 0.01;
opt = 0.25;
mu = x0;

logpx = target.logdensity(x0, target.inargs{:});
cnt = 0;
tic;
for it=1:Burn + T
%
   % draw the proposed sample
   epsilon = randn(1,n);
   if diagL == 1
     y = x + L.*(sqrt(delta)*epsilon);
   else
     y = x + (sqrt(delta)*epsilon)*L';
   end

   % Metropolis-Hastings ratio to accept or reject the samples
   logpy = target.logdensity(y, target.inargs{:});

   log_ratio = logpy - logpx;

   u = rand;
   accept = 0;
   if log(u)<log_ratio
     accept = 1;
     x = y;
     logpx = logpy;
   end

   acceptHist(it) = accept;

   % adapt the proposal during burning
   if (it <= Burn) && (adapt == 1)
     
      rho_Lt = rho_L/(1 + it/T0); % learning rate 
      
      mu = mu + rho_mu*(x - mu);
      if diagL == 1
         invLxm = (x-mu)./L;
         L = L + rho_Lt*(L.*(invLxm.*invLxm - 1)); 
      else
         invLxm = L\((x-mu)');
         L = L + rho_Lt*(L*(invLxm*invLxm' - eye(n)));
         L = L.*maskL; 
      end
                    
      % constraint for numerical stability
      if diagL == 1
         L(L<=1e-3)=1e-3;
      else
         keep = diag(L);
         keep(keep<=1e-3)=1e-3;
         L = L + (diag(keep - diag(L)));
      end

      % adapt global scaling to match a certain acceptance rate
      delta = delta + rho_delta*((accept - opt)/opt)*delta;
 
     if flagExtra == 1
        if diagL == 1
          invL = 1./L; 
          lambdas = trueSigma.*(invL.*invL);
        else  
          invL = L\eye(n);
          lambdas  = svd(trueSigma*(invL'*invL));
        end
        % compute the inhomogeneity factor b 
        bfactor(it) = (n*sum(lambdas))/( sum(sqrt(lambdas))^2 );
     end       
   elseif (mod(it,StoreEvery) == 0)
     cnt = cnt + 1;
     samples(cnt,:) = x;
   end
          
   logpxHist(it) = logpx;
%
end
timetaken = toc;
          
extraOutputs.logpxHist = logpxHist;
extraOutputs.acceptHist = acceptHist;
extraOutputs.L = L;
extraOutputs.delta = delta;
if flagExtra == 1
  extraOutputs.bfactor = bfactor;
end
extraOutputs.elapsed = timetaken; 
extraOutputs.accRate = mean(acceptHist(Burn+1:end));
