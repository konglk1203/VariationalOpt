function [g, out] = VariationalLieSGD(G, f_grad, hp)
    % Momentum gradient optimizer on the Stiefel manifold. (Algorithms in https://arxiv.org/pdf/2405.20390)

    % Inputs: 
    % G: Lie group object.
    % f_grad: objective function, takes in a element on G and output [function_value, gradient]
    % hp: hyperparameter
    % 
    % Outputs:
    % g: the local minimum return by the optimizer
    % out: a dict containing the training curve

    % How to adjust hp:
    % machine learning tasks: algo='nag_sc' or 'heavy_ball'; restart=false.
    % stochastic/noisy gradient: algo='nag_sc'; restart=false.
    % all other cases: algo='nag_c'; restart=false.
    if nargin < 2
        error('at least three inputs: [g, out] = VariationalLieSGD(G, f_grad, hp)');
    elseif nargin < 3
        hp = {};
    end

    % step size
    if ~isfield(hp, 'h');           hp.h = 0.1; end
    % initial value for the optimizer
    if ~isfield(hp, 'g_init');      hp.g_init = G.group_identity(); end
    % friction
    if ~isfield(hp, 'gamma');       hp.gamma = 1; end
    % which algorithm to use, should be in ['nag_sc', 'heavy_ball','nag_c', 'momentum_free']
    if ~isfield(hp, 'algo');        hp.algo = 'nag_c'; end
    % max iteration the algorithm runs
    if ~isfield(hp, 'max_iter');    hp.max_iter = 100; end
    % the algorithm will terminate when the Riemannian gradient is smaller than this value
    if ~isfield(hp, 'gtol');        hp.gtol = 0; end
    % whether to use restart scheme
    if ~isfield(hp, 'restart');     hp.restart = false; end
    % control whether to print during optimization
    if ~isfield(hp, 'verbose');     hp.verbose = true; end
    
    out={};
    out.loss_list=[];
    % initialize optimizer_status
    
    optimizer_status={};
    out.norm_grad_list=[];
    out.loss_list=[];
    % loop
    t_start=tic;
    g=hp.g_init;
    
    for iter = 1:hp.max_iter
        [f, euclidean_grad] = feval(f_grad, g);
        [g, optimizer_status] = update_VariationalLieSGD(G, g, euclidean_grad, optimizer_status, hp);
        if hp.restart
            optimizer_status = restart_VariationalLieSGD(X, optimizer_status, hp);
        end
        out.norm_grad_list(end+1) = optimizer_status.norm_grad;
        out.loss_list(end+1) = f;
        if optimizer_status.norm_grad<hp.gtol
            break
        end
    
        if hp.verbose
            fprintf('%3d %14.8e \n',iter, optimizer_status.norm_grad);
        end
        out.norm_grad = optimizer_status.norm_grad;
        
        disp(out.norm_grad)
    end
    if hp.verbose
        fprintf('total time: %.3e \n',toc(t_start));
    end
end

function [g, optimizer_status] = update_VariationalLieSGD(G, g, euclidean_grad, optimizer_status, hp)
    if ~isfield(optimizer_status, 'k')
        optimizer_status.k = 1;
    end
    if ~isfield(optimizer_status, 'xi')
        optimizer_status.xi = G.lie_algebra_zero();
    end
    if ~isfield(optimizer_status, 'trivialized_grad_last')
        optimizer_status.xi_last = G.lie_algebra_zero();
    end
    xi=optimizer_status.xi;
    h=hp.h;
    k=optimizer_status.k;
    if strcmp(algo, 'nag_sc') || strcmp(algo, 'heavy_ball')
        gamma=hp.gamma;
    elseif strcmp(algo, 'nag_c')
        gamma=3*(hp.gamma)/(3+h*k);
    elseif strcmp(algo, 'momentum_free')
        gamma=1/h;
    end
    manifold_grad=G.project_grad(g, euclidean_grad);
    trivialized_grad=G.trivialize(g, manifold_grad);
    norm_grad=G.norm(trivialized_grad);
    if strcmp(algo, 'nag_sc') || strcmp(algo, 'nag_c')
        if ~isfield(optimizer_status, 'trivialized_grad_last')
            optimizer_status.trivialized_grad_last=trivialized_grad;
        end
        trivialized_grad_last=optimizer_status.trivialized_grad_last;
        trivialized_grad_nag=(1-gamma*h)*(trivialized_grad-trivialized_grad_last)+trivialized_grad;
        optimizer_status.trivialized_grad_last=trivialized_grad;
        trivialized_grad=trivialized_grad_nag;
    end
    xi=xi*(1-gamma*h)-h*trivialized_grad;
    optimizer_status.xi=xi;
    g=G.multiplication(g, G.exp(h*xi));
    optimizer_status.k=k+1;
    optimizer_status.norm_grad=norm_grad;
end

function optimizer_status = restart_VariationalLieSGD(optimizer_status, hp)
    if G.norm(optimizer_status.xi)<G.norm(optimizer_status.xi)
        if hp.verbose
            fprintf('restart');
        end
        optimizer_status = rmfield(optimizer_status,'xi');
        optimizer_status = rmfield(optimizer_status,'trivialized_grad_last');
    end
end