function [g, out] = VariationalLieKLMC(G, f_grad, hp)
    % Kinetic Langevin Monte Carlo sampler on Lie group. (Algorithm 1 in https://proceedings.mlr.press/v247/kong24a/kong24a.pdf)
    % Given a Lie group $G$ and a potential function $f: G\to \mathbb{R}$, return samples from density
    % $\mu \propto\exp(-f(g))$

    % Inputs: 
    % G: Lie group object.
    % f_grad: potential function, takes in a element on G and output [function_value, gradient]
    % hp: hyperparameter
    % 
    % Outputs:
    % g: the last element in the MCMC chain
    % out: a dict containing the samples

    if nargin < 2
        error('at least three inputs: [g, out] = VariationalLieKLMC(G, f_grad, hp)');
    elseif nargin < 3
        hp = {};
    end

    % step size
    if ~isfield(hp, 'h');               hp.h = 0.1; end
    % friction
    if ~isfield(hp, 'gamma');           hp.gamma = 1; end
    % initial value for the optimizer
    if ~isfield(hp, 'g_init');          hp.g_init = G.group_identity(); end
    % max iteration the algorithm runs
    if ~isfield(hp, 'max_iter');        hp.max_iter = 1000; end
    % control whether to print during optimization
    if ~isfield(hp, 'verbose');         hp.verbose = true; end
    % number of warm up steps
    if ~isfield(hp, 'warmup_steps');    hp.warmup_steps = 100; end
    % if ~isfield(hp, 'mh_correction'); hp.mh_correction = false; end
    
    out={};
    % initialize optimizer_status
    
    optimizer_status={};
    out.samples={};
    % loop
    t_start=tic;
    g=hp.g_init;
    for iter = 1:(hp.max_iter+hp.warmup_steps)
        [f, euclidean_grad] = feval(f_grad, g);
        [g, optimizer_status] = update_VariationalLieSGD(G, g, euclidean_grad, optimizer_status, hp);
        if iter>hp.warmup_steps:
            out.samples=[out.samples, g];
        end
        if hp.verbose
            % fprintf('%3d %14.8e \n',iter, optimizer_status.norm_grad);
        end
    end
    if hp.verbose
        fprintf('total time: %.3e \n',toc(t_start));
    end
end

function [g, optimizer_status] = update_VariationalLieSGD(G, g, euclidean_grad, optimizer_status, hp)
    if ~isfield(optimizer_status, 'xi')
        optimizer_status.xi = G.lie_algebra_zero();
    end
    xi=optimizer_status.xi;
    h=hp.h;
    gamma=hp.gamma;
    manifold_grad=G.project_grad(g, euclidean_grad);
    trivialized_grad=G.trivialize(g, manifold_grad);
    % xi=xi*(1-gamma*h)-h*trivialized_grad+sqrt(1-exp(-2*gamma*h))*G.get_noise();
    xi=(1-gamma*h)*xi-h*trivialized_grad+sqrt(2*gamma*h)*G.get_noise();
    g=G.multiplication(g, G.exp(h*xi));

    optimizer_status.xi=xi;
end