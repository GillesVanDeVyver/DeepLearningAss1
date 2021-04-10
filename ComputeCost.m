function [cost,loss] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    lcross = -log(dot(Y,P));
    loss = sum(lcross)/size(X,2);
    nb_layers = size(W,2);
    regularisation_cost = 0;
    for l = 1:nb_layers
        W_squared = W{l}.^2;
        regularisation_cost = regularisation_cost + sum(W_squared(:));
    end
    cost = loss + lambda*regularisation_cost;
end