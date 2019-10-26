function [out, grad_W, grad_b] = log_ratio(epsilon, x, proposal, target)

beta= 2.56;

n = length(x);

sigmanet_x = netforward(proposal.sigmanet, x);
sigma_x = sigmanet_x{1}.Z;

y = x + sigma_x.*epsilon;

[logpy, grad_logpy] = target.logdensity(y, target.inargs{:});

% evaluate forward and backgward proposals q(y|x), q(x|y)
sigmanet_y = netforward(proposal.sigmanet, y);
sigma_y = sigmanet_y{1}.Z;
eps2 = epsilon.^2;
sigmax_div_y = sigma_x./sigma_y;
sigmax_div_y2 = sigmax_div_y.^2;
eps2_sigmax_div_y2 = eps2.*sigmax_div_y2;
log_q_y_x = - 0.5*n*log(2*pi) - sum(log(sigma_x)) - 0.5*sum( eps2, 2);
log_q_x_y = - 0.5*n*log(2*pi) - sum(log(sigma_y)) - 0.5*sum( eps2_sigmax_div_y2, 2);

out = logpy + log_q_x_y - log_q_y_x + beta*sum(log(sigma_x));

precondition1 = -1./sigma_y + eps2_sigmax_div_y2./sigma_y;
[grad_y_W, grad_y_b, grad_y] = netbackpropagation(sigmanet_y, precondition1, 0);
grad_y = grad_logpy + grad_y;
precondition2 = grad_y.*epsilon + 1./sigma_x - (eps2.*sigmax_div_y)./sigma_y + beta./sigma_x;
[grad_W, grad_b] = netbackpropagation(sigmanet_x, precondition2, 0);
for layer=length(sigmanet_x)-1:-1:1
  grad_W{layer} = grad_W{layer} + grad_y_W{layer};
  grad_b{layer} = grad_b{layer} + grad_y_b{layer};
end
