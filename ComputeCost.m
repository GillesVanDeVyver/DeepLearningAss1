function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    lcross = -log(dot(Y,P));
    J = sum(lcross)/size(X,2);
    W_squared = W.^2;
    J = J + lambda*sum(W_squared(:));
end