function P = EvaluateClassifier(X, W, b)
    s = W*X +b;
    denom = sum(exp(s),1);
    P = zeros(size(W,1),size(X,2));
    for i =1:size(s,2)
        P(:,i) = exp(s(:,i))/denom(i);
    end
end