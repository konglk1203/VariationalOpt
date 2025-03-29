% return a random Stiefel matrix with shape n-by-m

function Q = InitStiefelMatrix(n, m)
    Z = randn(n,m);
    [Q,R] = qr(Z,0);

    D = diag(R);
    Q = Q * diag(D ./ abs(D));

end