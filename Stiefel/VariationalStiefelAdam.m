function [X, out] = VariationalStiefelAdam(X_init, f_grad, hp)
    % Adam optimizer on the Stiefel manifold. (Algorithm 2 in https://arxiv.org/pdf/2205.14173)

    % Inputs: 
    % X_init: initial value for the optimizer, an n-by-m matrix on the Stiefel matrix.
    % f_grad: objective function, takes in a n-by-m matrix and output [function_value, gradient]
    % hp: hyperparameter
    % 
    % Outputs:
    % X: the local minimum return by the optimizer
    % out: a dict containing the training curve


    if nargin < 2
        error('at least 2 inputs: [x, out] = VariationalStiefelAdam(x, f_grad)');
    elseif nargin < 3
        hp = {};
    end
    
    % learning rate
    if ~isfield(hp, 'lr');              hp.lr = 1e-3; end
    % moving average parameter for first and second order momentum
    if ~isfield(hp, 'beta_1');          hp.beta_1 = 0.9; end
    if ~isfield(hp, 'beta_2');          hp.beta_2 = 0.99; end
    % epsilon for numerical stability
    if ~isfield(hp, 'epsilon');         hp.epsilon = 1e-8; end
    % control the Riemannian metric on the Stiefel metric. We recommend keeping this value as default setting. See Eq. 2 in https://arxiv.org/pdf/2205.14173 for details
    if ~isfield(hp, 'a');               hp.a = 0.5; hp.b=hp.a/(hp.a-1); end
    % control how to calculate matrix exponential. The algorithm is not sensitive to this. We recommend keeping this value as default setting.
    % should be in ['Cayley', 'MatrixExp', 'ForwardEuler']
    if ~isfield(hp, 'expm_method');     hp.expm_method = "ForwardEuler"; end
    % max iteration the algorithm runs
    if ~isfield(hp, 'max_iter');        hp.max_iter = 1000; end
    % the algorithm will terminate when the Riemannian gradient is smaller than this value
    if ~isfield(hp, 'gtol');            hp.gtol = 0; end
    % control whether to print during optimization
    if ~isfield(hp, 'verbose');         hp.verbose = true; end
    
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
        out.loss_list(end+1) = f_X;
        out.norm_grad_list(end+1) = optimizer_status.nrmG;        
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
    if isfield(optimizer_status, 'square')
        assert(optimizer_status.square==square)
    else
        optimizer_status.square=square;
    end
    square=optimizer_status.square;
    
    if ~isfield(optimizer_status, 'step')
        optimizer_status.step = 1;
    end
    if ~isfield(optimizer_status, 'Y')
        optimizer_status.Y = zeros(m,m);
    end
    if ~square
        if ~isfield(optimizer_status, 'V')
            optimizer_status.V = zeros(n,m);
        end
    end
    if ~isfield(optimizer_status, 'p_Y')
        optimizer_status.p_Y = zeros(m,m);
    end
    if ~square
        if ~isfield(optimizer_status, 'p_V')
            optimizer_status.p_V = zeros(n,m);
        end
    end
    if ~isfield(optimizer_status, 'Q')
        optimizer_status.Q = zeros(n,m);
    end
    
    
    Y=optimizer_status.Y;
    p_Y=optimizer_status.p_Y;
    if ~square
        V=optimizer_status.V;
        p_V=optimizer_status.p_V;
    end
    a=hp.a;
    b=hp.b;
    lr=hp.lr;
    expm_method=hp.expm_method;
    beta_1=hp.beta_1;
    beta_2=hp.beta_2;
    epsilon=hp.epsilon;
    step=optimizer_status.step;

    Xt_Xgrad=X'*X_grad;
    riemannian_grad=X_grad-X*(X'*X_grad);
    nrmG=norm(riemannian_grad,"fro");

    bias_correction_1 = 1 - beta_1^step;
    bias_correction_2 = 1 - beta_2^step;

    grad_Y=(1-b)/2*(Xt_Xgrad-Xt_Xgrad');
    if ~square
        grad_V=X_grad-X*Xt_Xgrad;
    end
    p_Y=p_Y*beta_2+(1-beta_2)*grad_Y.^2;
    if ~square
        p_V=beta_2*p_V+(1-beta_2)*grad_V.^2;
    end
    % Dynamics phi_2 (will be skipped when n=m)
    
    if ~square
        V=beta_1*V-(3*a-2)/2*V*Y-(1-beta_1)*grad_V;
    end
    % Dynamics phi_1
    Y=beta_1*Y-(1-beta_1)*grad_Y;
    
    
    denominator_Y=sqrt(p_Y/bias_correction_2)+epsilon;
    xi=lr/bias_correction_1*Y./denominator_Y;

    
    if expm_method=="Cayley"
        X=X*cayley(xi);
    elseif expm_method=="MatrixExp"
        X=X*expm(xi);
    elseif expm_method=="ForwardEuler"
        X=X+X*xi;
    else
        error("Error: expm_method should be in [''Cayley'', ''MatrixExp'', ''ForwardEuler'']")
    end

    % Dynamics phi_3 (will be skipped when n=m)
    if ~square
        denominator_V=sqrt(p_V/bias_correction_2)+epsilon;
        V_tilde=V./denominator_V-X*inv(X'*X)*(X'*(V./denominator_V));
        V_tilde=V_tilde/bias_correction_1;
        XVTV=X*(V_tilde'*V);
        X=X+lr*V_tilde*(X'*X);
        V=V-lr*XVTV;

    end
    X=X*((X'*X)^(-0.5));
    optimizer_status.step=step+1;
    optimizer_status.nrmG=nrmG;

    optimizer_status.Y=Y;
    optimizer_status.p_Y=p_Y;
    if ~square
        optimizer_status.V=V;
        optimizer_status.p_V=p_V;
    end

end