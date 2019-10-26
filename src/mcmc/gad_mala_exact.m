function [x, samples, extraOutputs] = gad_mala_exact(x0, target, L, beta, Burn, T, adapt, StoreEvery, rho_L, Optimizer)
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

[logpx, grad_logpx] = target.logdensity(x, target.inargs{:});

cnt = 0;
tic;
for it=1:Burn + T
%
  % draw the proposed sample
  epsilon = randn(1,n);
  if diagL == 1
    LTgrad_logpx = grad_logpx.*L;
    y = x + (0.5*LTgrad_logpx + epsilon).*L;
  else
    LTgrad_logpx = grad_logpx*L;
    y = x + (0.5*LTgrad_logpx + epsilon)*L';
  end

  % Metropolis-Hastings ratio to accept or reject the samples
  [logpy, grad_logpy, hessiany] = target.logdensity(y, target.inargs{:});

  % evaluate forward and backgward proposals q(y|x), q(x|y)
  % 
  if diagL == 1 
    eps2 = epsilon.^2;
    LTgrad_logpy = grad_logpy.*L;
    diff = 0.5*(LTgrad_logpx + LTgrad_logpy) + epsilon; 
    log_q_y_x = - 0.5*sum( eps2, 2);
    log_q_x_y = - 0.5*sum( diff.^2, 2); 
  else
    eps2 = epsilon.^2;
    LTgrad_logpy = grad_logpy*L;
    diff = 0.5*(LTgrad_logpx + LTgrad_logpy) + epsilon; 
    log_q_y_x = - 0.5*sum( eps2, 2);
    log_q_x_y = - 0.5*sum( diff.^2, 2); 
  end

  log_ratio = logpy - logpx + log_q_x_y - log_q_y_x;
  
  u = rand;
  accept = 0;
  if log(u)<log_ratio
   accept = 1;
  end

  acceptHist(it) = accept;

  % adapt the proposal during burning
  if (it <= Burn) && (adapt == 1)
   if log_ratio < 0
     term2 = grad_logpx + grad_logpy;   
     if diagL == 1
       L_gradlogpx_eps = LTgrad_logpx + epsilon; 
       grad_L = grad_logpy.*L_gradlogpx_eps...
             - 0.5*(diff.*term2 + ((diff.*L)*hessiany).*L_gradlogpx_eps) + beta*(1./L);
     else
       keep = (diff*L')*hessiany;
       LTkeep = keep*L;
       grad_L = 0.5*(grad_logpx'*LTgrad_logpy + grad_logpy'*LTgrad_logpx) + grad_logpy'*epsilon...
              - 0.5*(term2'*diff + (0.5*(grad_logpx'*LTkeep + keep'*LTgrad_logpx) + keep'*epsilon ));          
       grad_L = grad_L.*maskL + beta*diag(1./diag(L));
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

  if accept == 1
    x = y;
    logpx = logpy;
    grad_logpx = grad_logpy;
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
