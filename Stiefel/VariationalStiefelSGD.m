function [X, out] = VariationalStiefelSGD(X_init, f_grad, hp)
    % Momentum gradient optimizer on the Stiefel manifold. (Algorithm 1 in https://arxiv.org/pdf/2205.14173)

    % Inputs: 
    % X_init: initial value for the optimizer, an n-by-m matrix on the Stiefel matrix.
    % f_grad: objective function, takes in a n-by-m matrix and output [function_value, gradient]
    % hp: hyperparameter
    % 
    % Outputs:
    % X: the local minimum return by the optimizer
    % out: a dict containing the training curve

    % How to adjust hp:
    % machine learning tasks: algo='nag_sc' or 'heavy_ball'; restart=false.
    % stochastic/noisy gradient: algo='nag_sc'; restart=false.
    % all other cases: algo='nag_c'; restart=false.


    if nargin < 2
        error('at least 2 inputs: [x, out] = VariationalStiefelSGD(X_init, f_grad)');
    elseif nargin < 3
        hp = {};
    end

    
    % step size
    if ~isfield(hp, 'h');            hp.h = 0.1; end
    % friction
    if ~isfield(hp, 'gamma');        hp.gamma = 0.01; end
    % which algorithm to use, should be in ['nag_sc', 'heavy_ball','nag_c', 'momentum_free']
    if ~isfield(hp, 'algo');         hp.algo = 'nag_c'; end
    % control the Riemannian metric on the Stiefel metric. We recommend keeping this value as default setting. See Eq. 2 in https://arxiv.org/pdf/2205.14173 for details
    if ~isfield(hp, 'a');            hp.a = 0.5; hp.b=hp.a/(hp.a-1); end
    % control how to calculate matrix exponential. The algorithm is not sensitive to this. We recommend keeping this value as default setting.
    % should be in ['Cayley', 'MatrixExp', 'ForwardEuler']
    if ~isfield(hp, 'expm_method');  hp.expm_method = "ForwardEuler"; end
    % max iteration the algorithm runs
    if ~isfield(hp, 'max_iter');     hp.max_iter = 1000; end
    % the algorithm will terminate when the Riemannian gradient is smaller than this value
    if ~isfield(hp, 'gtol');         hp.gtol = 0; end
    % whether to use restart scheme
    if ~isfield(hp, 'restart');      hp.restart = true; end
    % control whether to print during optimization
    if ~isfield(hp, 'verbose');      hp.verbose = false; end
    
    out=[];
    out.loss_list=[];
    % initialize optimizer_status
    X=X_init;
    optimizer_status=[];
    out.norm_grad_list=[];
    % loop
    for iter = 1:hp.max_iter
        [f_X, X_grad] = feval(f_grad, X);
    
        [X, optimizer_status] = update_VariationalStiefelSGD(X, X_grad, optimizer_status, hp);
        if hp.restart
            optimizer_status = restart_VariationalStiefelSGD(X, optimizer_status, hp);
        end
        out.loss_list(end+1) = f_X;
        out.norm_grad_list(end+1) = optimizer_status.nrmG;
        if optimizer_status.nrmG<hp.gtol
            break
        end
        if hp.verbose
            fprintf('%3d %14.8e \n',iter, optimizer_status.nrmG);
        end
        out.nrmG = optimizer_status.nrmG;
    end 
end

function [X, optimizer_status] = update_VariationalStiefelSGD(X, X_grad, optimizer_status, hp)
    [n,m]=size(X);
    assert(n>=m);
    square=(n==m);

    % initialize optimizer
    if isfield(optimizer_status, 'square')
        assert(optimizer_status.square==square)
    else
        optimizer_status.square=square;
    end
    square=optimizer_status.square;
    
    if ~isfield(optimizer_status, 'k')
        optimizer_status.k = 1;
    end
    if ~isfield(optimizer_status, 'Y')
        optimizer_status.Y = zeros(m,m);
    end
    if ~square
        if ~isfield(optimizer_status, 'V')
            optimizer_status.V = zeros(n,m);
        end
    end
    if ~isfield(optimizer_status, 'Q')
        optimizer_status.Q = zeros(n,m);
    end
    if ~isfield(optimizer_status, 'Q_last')
        optimizer_status.Q_last = zeros(n,m);
    end

    Y=optimizer_status.Y;
    if ~square
        V=optimizer_status.V;
    end
    a=hp.a;
    b=hp.b;
    h=hp.h;
    k=optimizer_status.k;
    algo=hp.algo;
    if strcmp(algo, 'nag_sc') || strcmp(algo, 'heavy_ball')
        gamma=hp.gamma;
    elseif strcmp(algo, 'nag_c')
        gamma=3*(hp.gamma)/(3+h*k);
    elseif strcmp(algo, 'momentum_free')
        gamma=1/h;
    end
    
    expm_method=hp.expm_method;
    Xt_Xgrad=X'*X_grad;
    grad_Y=(1-b)/2*(Xt_Xgrad-Xt_Xgrad');
    riemannian_grad=X_grad-(1+b)/2*X*(X'*X_grad)-(1-b)/2*X*(X_grad'*X);
    nrmG=norm(riemannian_grad,"fro");

    if ~square
        grad_V=X_grad-X*Xt_Xgrad;
        if strcmp(algo, 'nag_sc') || strcmp(algo, 'nag_c')
            if ~isfield(optimizer_status, 'grad_V_last')
                optimizer_status.grad_V_last=grad_V;
            end
            grad_V_last=optimizer_status.grad_V_last;
            grad_V_NAG=(1-gamma*h)*(grad_V-grad_V_last)+grad_V;
            optimizer_status.grad_V_last=grad_V;
            grad_V=grad_V_NAG-X*(X'*grad_V_NAG);
        end
        % Dynamics phi_2 (will be skipped when n=m)
        V=V*(1-gamma*h)-(3*a-2)/2*h*V*Y-h*grad_V;
    end
    % Dynamics phi_1
    Y=Y*(1-gamma*h)-h*grad_Y;
    optimizer_status.Y=Y;
    
    if expm_method=="Cayley"
        X=X*cayley(h*Y);
    elseif expm_method=="MatrixExp"
        X=X*expm(h*Y);
    elseif expm_method=="ForwardEuler"
        X=X+h*X*Y;
    else
        error("Error: expm_method should be in [''Cayley'', ''MatrixExp'', ''ForwardEuler'']")
    end
    % Dynamics phi_3 (will be skipped when n=m)
    if ~square
        VTV=V'*V;
        XVTV=X*VTV;
        X=X+h*V*(X'*X);
        V=V-h*XVTV;
        optimizer_status.V=V;
    end
    X=X*((X'*X)^(-0.5));
    optimizer_status.k=k+1;
    optimizer_status.nrmG=nrmG;

    optimizer_status.Q_last=optimizer_status.Q;
    if ~square
        optimizer_status.Q=X*Y+V;
    else
        optimizer_status.Q=X*Y;
    end
end

function optimizer_status = restart_VariationalStiefelSGD(X, optimizer_status, hp)
    if norm(optimizer_status.Q)<norm(optimizer_status.Q_last)
        if hp.verbose
            fprintf('restart');
        end
        hp.k=1;
        [n,m]=size(X);
        optimizer_status.Y = zeros(m,m);
        if isfield(optimizer_status, 'V')
            optimizer_status.V = zeros(n,m);
        end
        
        optimizer_status.Q= zeros(m,m);
        optimizer_status.Q_last = zeros(n,m);
        if isfield(optimizer_status, 'grad_Y_last')
            optimizer_status = rmfield(optimizer_status,'grad_Y_last');
        end
        if isfield(optimizer_status, 'grad_V_last')
            optimizer_status = rmfield(optimizer_status,'grad_V_last');
        end
    end
end