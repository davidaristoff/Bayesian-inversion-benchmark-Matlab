%load forward solver and gradient solver
load precomputations.mat

for i=1:9

    %open test input "theta" file
    formatSpec = '%f';
    fileID = fopen(['testing/input.' num2str(i) '.txt']);
    theta_test = fscanf(fileID,formatSpec);

    %open associated output "z" file
    fileID = fopen(['testing/output.' num2str(i) '.z.txt']);
    z_test = fscanf(fileID,formatSpec);

    %test forward solver (with and without gradients)
    [z,dz] = forward_solver_with_gradient_(theta_test);

    %display relative errors on z
    disp(['file number ' num2str(i) ... 
        ' relative errors on z using forward_solver_with_gradient = ...'])
    norm(z-z_test)/norm(z_test)

    %now check accuracy of derivatives
    del = 10^(-6);    %finite difference (FD) parameter
    rel_errors = zeros(64,1);    %vector of relative errors in dz
    for j=1:64
        theta = theta_test;
        theta(j) = theta(j) + del;
        [z_,~] = forward_solver_with_gradient_(theta);
        dzj_fin_diff = (z_-z)/del;    %FD approximation of dz/dthetaj
        dzj_exact = dz(:,j);    %forward solver value for dz/dthetaj
        rel_errors(j) = norm(dzj_fin_diff-dzj_exact)/norm(dzj_exact);
    end
    
    %display relative errors on dz
    disp(['file number ' num2str(i) ' max relative error on dz = ...'])
    max(rel_errors)

end

%check forward solver runtime in milliseconds
disp('time for forward solver to run 1000 times = ...')
tic
for i=1:1000
    theta = normrnd(1,0.1,[64 1]);
    z = forward_solver_(theta);
end
toc

%check forward solver with gradients runtime in milliseconds
disp('time for forward solver with gradients to run 1000 times = ...')
tic
for i=1:1000
    theta = normrnd(1,0.1,[64 1]);
    [z,dz] = forward_solver_with_gradient_(theta);
end
toc
