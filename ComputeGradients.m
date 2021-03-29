function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    n = size(X,2);
    G_batch = -(Y-P);
    grad_W = 1/n*G_batch*X.' + 2*lambda*W;
    grad_b = 1/n*G_batch*ones(n,1);
end