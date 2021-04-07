function P = EvaluateClassifier(X, W, b)
    W_mat = cell2mat(W);
    b_mat = cell2mat(b);
    s = W_mat*X +b_mat;
    denom = sum(exp(s),1);
    P = zeros(size(W_mat,1),size(X,2));
    for i =1:size(s,2)
        P(:,i) = exp(s(:,i))/denom(i);
    end
end