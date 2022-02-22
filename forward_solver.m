%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% forward solver function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%theta = current 8x8 parameter matrix
%lbl = cell labeling function
%A_loc = matrix of local contributions to A
%Id = Identity matrix of size 128x128
%boundaries = labels of boundary cells
%b = right hand side of linear system (AU = b)
%M = measurement matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%z = vector of measurements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function z = forward_solver(theta,lbl,A_loc,Id,boundaries,b,M)

%initialize matrix A for FEM linear solve, AU = b
A = zeros(33^2,33^2);

%loop over cells to build A
for i=0:31
    for j=0:31   %build A by summing over contribution from each cell

        %find local coefficient in 8x8 grid
        theta_loc = theta(floor(i/4)+1,floor(j/4)+1);

        %update A by including contribution from cell (i,j)
        dof = [lbl(i,j),lbl(i,j+1),lbl(i+1,j+1),lbl(i+1,j)];
        A(dof,dof) = A(dof,dof) + theta_loc*A_loc;
    end
end

%enforce boundary condition
A(boundaries,:) = Id(boundaries,:);
A(:,boundaries) = Id(:,boundaries);

%sparsify A
A = sparse(A);

%solve linear equation for coefficients, U
U = A\b;

%get new z values
z = M*U;
