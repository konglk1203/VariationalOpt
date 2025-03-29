
% This is an example for the usage of Stiefel optimizers (VariationalStiefelSGD and VariationalStiefelAdam)
% the goal is to calculate the largest m eigenvalues of a n-by-n symmetric matrix A
% we formulate it as a optimization problem on Stiefel manifold, i.e., 
% min tr(X^T A X N), s.t., X is a Stiefel matrix, where N=diag(1, ..., m)
% and we solve it with the Stiefel optimizers

% More details and theoretical analysis: 
% Sec Q in https://arxiv.org/pdf/2205.14173
%% define the eigenvalue decomposition problem
n=10;
m=5;
kappa=100; % condition number for the eigenvalue decomposition problem. Larger kappa means the problem is harder.

eig_vals=generate_eig_value_artificial_conditional_number(n, m, kappa);
[A, min_val, mu, L, X_sol]=lev_problem(eig_vals, m);

X_init=InitStiefelMatrix(n,m);



noise=randn(size(A))*0.01;
X_init=expm(noise-noise')*X_sol;

%% hyperparameter (feel free to adjust it!)
hp={};
hp.h=1/sqrt(2*L);
hp.gamma=2*sqrt(mu)/(1-sqrt(mu)*hp.h);
hp.max_iter=1000;
hp.use_update=true;
hp.restart = true;

%% optimization
[X_nag, out_nag]=VariationalStiefelSGD(X_init, @(x)f_grad(x,A, min_val), hp);

% [X_adam, out_adam]=VariationalStiefelAdam(X_init, @(x)f_grad(x,A, min_val));


figure(1)
plot(out_nag.loss_list)
legend('NAG')
set(gca, 'YScale', 'log')

figure(2)
plot(out_nag.norm_grad_list)
legend('NAG')
set(gca, 'YScale', 'log')


%% helper functions
function eig_vals=generate_eig_value_artificial_conditional_number(n, m, kappa)
    assert(m>=2)
    assert(n>=m)
    assert(kappa>m*(m-1))
    eig_vals=1:n;
    eig_vals(end)=kappa/(m-1)+n-m;
end

function [A, min_val, mu, L, X_sol]=lev_problem(eig_vals, m)
    n=length(eig_vals);
    eig_vecs=randn(n, n);
    eig_vecs=orth(eig_vecs);
    A=eig_vecs*diag(eig_vals)*eig_vecs';
    min_val=sum(eig_vals(n-m+1:end).*(1:m), "all");
    L=(eig_vals(end)-eig_vals(n-m+1))*(m-1);
    mu=min(eig_vals(n-m+2:end)-eig_vals(n-m+1:end-1));
    X_sol=eig_vecs(:, n-m+1:end);
end 

function loss=eig_val_decomp_loss(A, X)
    [n,m]=size(X);
    D=diag(1:m);
    loss=-sum(X.*(A*X*D), "all");
end

function [f_X, grad_f]=f_grad(x,A, min_val)
    f_X=eig_val_decomp_loss(A, x)+min_val;
    m=size(x, 2);
    grad_f=-A*x*diag(1:m);
end