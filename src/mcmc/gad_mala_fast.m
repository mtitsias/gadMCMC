function [x, samples, extraOutputs] = ada_mala_fast(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer)
% 

if nargin < 8
  StoreEvery = 1;
  Optimizer = 1; % RMSprop optimizer
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
opt = 0.55;
TT = 1;
kappa0 = 0.1;
Gt_L = 0.01*ones(size( L ));

% adam optimizer settings
alpha = 0.0002;
beta1 = 0.9; 
beta2 = 0.999;
epsilon0 = 1e-8; 
m0 = zeros(size( L ));
v0 = zeros(size( L ));

tic;
[logpx, grad_logpx] = target.logdensity(x, target.inargs{:});
if diagL == 1
   LTgrad_logpx = grad_logpx.*L;
else
   LTgrad_logpx = grad_logpx*L;
end
cnt = 0;
for it=1:Burn + T
%
  % draw the proposed sample
  epsilon = randn(1,n);
  if diagL == 1
    y = x + (0.5*LTgrad_logpx + epsilon).*L;
  else
    half_LTgrad_logpx_epsilon = 0.5*LTgrad_logpx + epsilon;
    y = x + half_LTgrad_logpx_epsilon*L';  % --- O(n^2) operation
  end

  % evaluate new target and gradient 
  [logpy, grad_logpy] = target.logdensity(y, target.inargs{:});

  % evaluate forward and backgward proposals q(y|x), q(x|y) 
  if diagL == 1 
    eps2 = epsilon.^2;
    LTgrad_logpy = grad_logpy.*L;
    diff = 0.5*(LTgrad_logpx + LTgrad_logpy) + epsilon; 
    log_q_y_x = - 0.5*sum( eps2, 2);
    log_q_x_y = - 0.5*sum( diff.^2, 2); 
  else
    eps2 = epsilon.^2;
    LTgrad_logpy = grad_logpy*L;   % -- O(n^2) operation
    %diff = 0.5*(LTgrad_logpx + LTgrad_logpy) + epsilon;
    half_LTgrad_logpy = 0.5*LTgrad_logpy;
    diff =  half_LTgrad_logpx_epsilon + half_LTgrad_logpy;
    log_q_y_x = - 0.5*sum( eps2, 2);
    log_q_x_y = - 0.5*sum( diff.^2, 2); 
  end

  log_ratio = logpy - logpx + log_q_x_y - log_q_y_x;
  
  % Decide to accept or reject  
  u = rand;
  accept = 0;
  if log(u)<log_ratio
   accept = 1;
  end
  acceptHist(it) = accept;

  % Adapt the proposal during burning
  if (it <= Burn) && (adapt == 1)
   if log_ratio < 0 
     if diagL == 1
       term2 = grad_logpx + grad_logpy;
       L_gradlogpx_eps = LTgrad_logpx + epsilon; 
       grad_L = grad_logpy.*L_gradlogpx_eps...
             - 0.5*(diff.*term2) + (beta./L);
     else
       term2 = 0.5*(grad_logpx - grad_logpy); 
       grad_L = - term2'*(half_LTgrad_logpx_epsilon - half_LTgrad_logpy); % O(n^2) operation
      
       grad_L(1:1+n:end) = grad_L(1:1+n:end) + beta./(diag(L)');
       grad_L = tril(grad_L);   
     end
   else
     if diagL == 1
       grad_L = beta./L;
     else
       grad_L = diag(beta./diag(L));
     end  
   end

   if Optimizer == 1 % RMSprop   
      kappa = kappa0;
      if(it==1)
        kappa = 1;
      end
      Gt_L = kappa*(grad_L.^2) + (1-kappa)*Gt_L;
      %rho_Lt = rho_L/(1 + it/T0); % learning rate
      L = L + rho_L*(grad_L./(TT+sqrt( Gt_L )));
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
      if min(keep) < 1e-3 
        keep(keep<=1e-3)=1e-3;
        L(1:1+n:end) = L(1:1+n:end) + (keep - diag(L))';
        %L = L + (diag(keep - diag(L)));
      end
   end
    
   % update the x-value precomputations that depend on L
   % only if y is rejected, since if accepted it will 
   % be updated anyway
   if accept == 0
     if diagL == 1
       LTgrad_logpx = grad_logpx.*L;
     else
       LTgrad_logpx = grad_logpx*L; % O(n^2) operation 
     end
   end

   % update also the hyperparameter beta to match the desirable accept rate
   beta = beta + rho_beta*(accept - opt)*beta;
   beta(beta<0.0001)=0.0001;

   % stochastic value of the lower bound (never used... to cooment out)
   if diagL == 1
     lowerboundHist(it) = min(0,log_ratio) + beta*sum(log(L));
   else
     lowerboundHist(it) = min(0,log_ratio) + beta*sum(log(diag(L)));
   end
    
  elseif (mod(it,StoreEvery) == 0)
   cnt = cnt + 1;
   samples(cnt,:) = x;
  end

  if accept == 1
    x = y;
    logpx = logpy;
    grad_logpx = grad_logpy;
    LTgrad_logpx = LTgrad_logpy;
  end
  
  logpxHist(it) = logpx;

end
timetaken = toc;

extraOutputs.logpxHist = logpxHist;
extraOutputs.acceptHist = acceptHist;
extraOutputs.lowerboundHist = lowerboundHist;
extraOutputs.L = L;
extraOutputs.beta = beta;
extraOutputs.elapsed = timetaken;
extraOutputs.accRate = mean(acceptHist(Burn+1:end));
