%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% compute log probability, log pi %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%theta = current 8x8 parameter matrix
%z = current vector of measurements
%z_hat = vector of "exact" measurements
%sig = standard deviation parameter in likelihood
%sig_pr = standard deviation parameter in prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%log_pi = logarithm of posterior probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad_log_pi = grad_log_probability(theta,z,dz,z_hat,sig,sig_pr)

%compute log likelihood
grad_log_L = -dz'*(z-z_hat)/(sig^2);

%compute log prior
grad_log_pi_pr = -(log(theta)./theta)/(sig_pr^2);

%compute log posterior
grad_log_pi = grad_log_L + grad_log_pi_pr;
