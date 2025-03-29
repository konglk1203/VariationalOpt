%% construct the Lie group
n=10;
G=SOn(n);

% define the density function f(x)=-10 x(1,1)^2$ for $x\in SO(10)
% Note that given f defined here, the density function we are sampling from is \mu\propto \exp(-f)$.
function [f_X, grad_f]=f_grad(x)
    f_X=1-10*x(1,1)^2;
    grad_f=zeros(size(x));
    grad_f(1,1)=-20*x(1,1);
end

%% hyperparameter
hp={};
hp.h=0.1;
hp.gamma=1;
hp.max_iter=1000;

%% sampling
[g, out]=VariationalLieKLMC(G, @f_grad, hp);


%% plot
[f_list, grad_list] = cellfun(@f_grad, out.samples, 'UniformOutput',false);

plot_val=cellfun(@(x)x(1,1),out.samples);
figure(1)
histogram(plot_val, 'Normalization','pdf');
