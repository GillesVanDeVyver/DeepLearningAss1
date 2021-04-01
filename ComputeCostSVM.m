function J = ComputeCostSVM(X, Y, W, b, lambda)
    n = size(X,2);
    s = W*X+b;
    sy = sum(Y.*s, 1);
    Ynot = double(~Y);
    loss = Ynot.*(s-sy+1);
    loss = loss.*(loss>0);
    sum_loss = sum(loss);
    J = sum(sum_loss)/size(X,2);
    W_squared = W.^2;
    J = J + lambda*sum(W_squared(:));
end