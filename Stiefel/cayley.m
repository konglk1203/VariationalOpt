% Cayley map as an approximation of matrix exponential.

function exp_xi=cayley(xi)
    id=eye(size(xi));
    exp_xi=(id-xi/2)\(id+xi/2);
end