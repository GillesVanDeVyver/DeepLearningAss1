function [grad_W, grad_b] = ComputeGradientsSVM(X, Y, b, W, lambda)
    n = size(X,2);
    s = W*X+b;
    sy = sum(Y.*s, 1);
    Ynot = double(~Y);
    loss = Ynot.*(s-sy+1);
    miss_classification = double(loss>0);
    miss_sum= sum(miss_classification);
    grady = -miss_sum.*Y;
    g = miss_classification+grady;
    grad_W = 1/n*g*X.' +2*lambda*W;
    grad_b = 1/n*g*ones(n,1);
end