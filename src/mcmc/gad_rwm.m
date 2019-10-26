function [x, samples, extraOutputs] = gad_rwm(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer)

if nargin < 8
  StoreEvery = 1;
  Optimizer = 1;
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
lowerboundHist = zeros(1,Burn);
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
rho_beta = 0.02;
T0 = round(Burn/4); 
opt = 0.25;
TT = 1;
kappa0 = 0.1;
Gt_L = 0.01*ones(size( L ));

% adam optimizer settings
alpha = 0.00005;
beta1 = 0.9; 
beta2 = 0.999;
epsilon0 = 1e-8; 
m0 = zeros(size( L ));
v0 = zeros(size( L ));

logpx = target.logdensity(x0, target.inargs{:});
cnt = 0;
tic;
for it=1:Burn + T
%
   % draw the proposed sample
   epsilon = randn(1,n);
   if diagL == 1
     y = x + (L.*epsilon);
   else
     y = x + (epsilon*L');
   end

   % Metropolis-Hastings ratio to accept or reject the samples
   [logpy, grad_logpy] = target.logdensity(y, target.inargs{:});

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
       if log_ratio < 0
        if diagL == 1
          grad_L = (grad_logpy.*epsilon) + beta*(1./L);
        else
          grad_L = (grad_logpy'*epsilon).*maskL + beta*diag(1./diag(L));
        end
      else
        if diagL == 1
          grad_L = beta*(1./L);
        else
          grad_L = beta*diag(1./diag(L));
        end
      end
          
      if Optimizer == 1 % RMSprop   
         kappa = kappa0;
         if(it==1)
           kappa = 1;
         end
         Gt_L = kappa*(grad_L.^2) + (1-kappa)*Gt_L;
         %rho_Lt = rho_L/(1 + it/T0); % learning rate
         L = L + rho_L*grad_L./(TT+sqrt( Gt_L ));
      elseif Optimizer == 2 % Robins Monroe
         rho_Lt = rho_L/(1 + it/T0); % learning rate
         L = L + rho_Lt*grad_L;
      elseif Optimizer == 3 % Adam 
         m0 = beta1*m0 + (1-beta1)*grad_L;
         v0 = beta2*v0 + (1-beta2)*(grad_L.^2);
         alphat = alpha*sqrt(1-beta2^it)/(1-beta1^it);
         L = L + alphat*(m0./(sqrt(v0) + epsilon0));
      end 

      % constraint for numerical stability
      if diagL == 1
         L(L<=1e-3)=1e-3;
      else
         keep = diag(L);
         keep(keep<=1e-3)=1e-3;
         L = L + (diag(keep - diag(L)));
      end
          
      % update also the hyperparameter beta
      beta = beta + rho_beta*(accept - opt)*beta;
      beta(beta<0.0001)=0.0001;
          
      % stochastic value of the lower bound
      if diagL == 1
        lowerboundHist(it) = min(0,log_ratio) + beta*sum(log(L));
      else
        lowerboundHist(it) = min(0,log_ratio) + beta*sum(log(diag(L)));
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
extraOutputs.lowerboundHist = lowerboundHist;
extraOutputs.L = L;
extraOutputs.beta = beta;
extraOutputs.elapsed = timetaken; 
extraOutputs.accRate = mean(acceptHist(Burn+1:end));
