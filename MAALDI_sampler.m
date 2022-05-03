%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run ALDI sampler to estimate posterior distribution %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define number of chains, chain length, and lag time
N = input('number of particles: ');
N_L = input('length of each Markov chain: ');
lag = input('lag time for measurements: ');
dt = input('time step: ');
num_workers = input('number of parallel workers: ');
L = N_L/lag;

%start parallel workers
parpool(num_workers)

%load precomputations
load precomputations.mat

%define data matrix of mean theta values
data = zeros(64,L);

%define initial m = log(theta) values
ms = normrnd(0,0.1,[64 N]);

%initialize gradient values
gradients = zeros(64,N);

%initialize running mean values of theta
theta_mean = zeros(64,1);

tic

for k=1:L

    disp('percent complete = ...')
    k/L
    
    for l=1:lag

        %compute particle mean vector and covariance matrix
        ms_mean = mean(ms,2);
        sqrt_particle_covar = (ms-ms_mean)/sqrt(N);
        particle_covar = sqrt_particle_covar*sqrt_particle_covar';

        %update running mean values of theta
        theta_mean = theta_mean + exp(ms_mean);

        %initialize proposal and acceptance probability
        ms_prop = zeros(64,N);
        log_accept = zeros(N,1);

        %update particle positions
        parfor n=1:N

            %extract current particle
            m = ms(:,n);

            %find gradient for current particle
            [z,dz] = forward_solver_with_gradient_(exp(m));
            gradient = grad_log_probability_(m,z,dz);
    
            %define proposal associated to current particle
            m_prop = m + ...
                         particle_covar*gradient*dt + ... 
                         ((d+1)/N)*(m-ms_mean)*dt + ...
                         sqrt(2*dt)*sqrt_particle_covar*normrnd(0,1,[N 1]);

            %compute gradient of m_prop
            [z_prop,dz_prop] = forward_solver_with_gradient_(exp(m_prop));
            gradient_prop = grad_log_probability_(m_prop,z_prop,dz_prop); 

            %compute posterior log probability of m and m_prop
            log_pi_prop = log_probability_(exp(m_prop),z_prop);
            log_pi = log_probability_(exp(m),z);

            %update log acceptance probability
            log_accept(n) = log_pi_prop-log_pi ...
                         -norm(m-m_prop-dt*gradient_prop)^2/(4*dt) ...
                         +norm(m_prop-m-dt*gradient)^2/(4*dt);

            %update proposal
            ms_prop(:,n) = m_prop;

        end

        %accept or reject proposal
        sum(log_accept)
        if rand < exp(sum(log_accept))
            ms = ms_prop;
            disp('accept')
        end

    end
    
    %update data matrix
    data(:,k) = exp(ms_mean);

end

%normalize running mean values of theta
theta_mean = theta_mean/N_L;

toc

%shut down parallel pool
poolobj = gcp('nocreate');
delete(poolobj);

%save data to Matlab workspace, labeled by N and N_L
save (['MAALDI_data_N_' num2str(N) '_N_L_' num2str(N_L) '_dt_' num2str(dt) '.mat'])
