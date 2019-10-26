function [out, grad_W, grad_b, grad_L] = log_ratio(epsilon, x, proposal, target)

beta= 2.56;

n = length(x);

munet_x = netforward(proposal.munet, x);
mu_x = munet_x{1}.Z;

y = x + mu_x + proposal.L.*epsilon;


% Metropolis-Hastings ratio to accept or reject the samples
%
[logpy, grad_logpy] = target.logdensity(y, target.inargs{:});

% evaluate forward and backgward proposals q(y|x), q(x|y)
%
munet_y = netforward(proposal.munet, y);
mu_y = munet_y{1}.Z;
eps2 = epsilon.^2;
x_y_muy_L = (x - y - mu_y)./proposal.L;
x_y_muy_L2 = x_y_muy_L.^2;
log_q_y_x = - 0.5*sum( eps2, 2);
log_q_x_y = - 0.5*sum( x_y_muy_L2, 2);

out = logpy + log_q_x_y - log_q_y_x + beta*sum(log(proposal.L));

precondition1 = x_y_muy_L./proposal.L;
[grad_y_W, grad_y_b, grad_y] = netbackpropagation(munet_y, precondition1, 1);
grad_y = grad_logpy + precondition1 + grad_y;
[grad_W, grad_b] = netbackpropagation(munet_x, grad_y, 1);
for layer=length(munet_x)-1:-1:1
   grad_W{layer} = grad_W{layer} + grad_y_W{layer};
   grad_b{layer} = grad_b{layer} + grad_y_b{layer};
end

grad_L = grad_y.*epsilon + x_y_muy_L2./proposal.L + beta*(1./proposal.L);
