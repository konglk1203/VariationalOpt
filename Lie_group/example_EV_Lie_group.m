% This is an example for the usage of Lie group optimizer (VariationalLieSGD)
% the goal is to calculate the eigenvalue decomposition of a n-by-n symmetric matrix A
% we formulate it as a optimization problem on Stiefel manifold, i.e., 
% min tr(X^T A X N), s.t., X is a orthogonal matrix, where N=diag(1, ..., n)
% and we solve it with the Lie group optimizer optimizers

% More details and theoretical analysis: 
% Sec 6 in https://arxiv.org/pdf/2405.20390
%% define the eigenvalue decomposition problem
n=100;
kappa=20000; % condition number for the eigenvalue decomposition problem. Larger kappa means the problem is harder.



eig_vals=generate_eig_value_artificial_conditional_number(n, kappa);
[A, min_val, mu, L, X_sol]=lev_problem(eig_vals);

function [f_X, grad_f]=f_grad(x,A, min_val)
    f_X=eig_val_decomp_loss(A, x)+min_val;
    m=size(x, 2);
    grad_f=-A*x*diag(1:m);
end

%% construct the Lie group
G=SOn(n);

%% hyperparameter (feel free to adjust it!)
hp={};
hp.h=1/sqrt(2*L);
hp.gamma=2*sqrt(mu)/L;
hp.algo='nag_sc';
hp.restart=false;
hp.max_iter=10;
hp.max_iter=100;

%% optimization
[g, out]=VariationalLieSGD(G, @(x)f_grad(x,A, min_val), hp);


figure(1)
plot(out_nag.loss_list)
legend('NAG')
set(gca, 'YScale', 'log')

figure(2)
plot(out_nag.norm_grad_list)
legend('NAG')
set(gca, 'YScale', 'log')

%% helper functions
function eig_vals=generate_eig_value_artificial_conditional_number(n, kappa)
    assert(kappa>(n-1)^2)
    eig_vals=1:n;
    eig_vals(end)=kappa/(n-1);
end


function [A, min_val, mu, L, X_sol]=lev_problem(eig_vals)
    n=length(eig_vals);
    eig_vecs=randn(n, n);
    eig_vecs=orth(eig_vecs);
    A=eig_vecs*diag(eig_vals)*eig_vecs';
    min_val=sum(eig_vals.*(1:n), "all");
    L=(eig_vals(end)-eig_vals(1))*(n-1);
    mu=min(eig_vals(2:end)-eig_vals(1:end-1));
    X_sol=eig_vecs;
end




function loss=eig_val_decomp_loss(A, X)
    n=size(X,1);
    D=diag(1:n);
    loss=-sum(X.*(A*X*D), "all");
end