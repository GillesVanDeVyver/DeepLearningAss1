function [P,h,s] = EvaluateClassifier(X, W, b)
    nb_layers = size(W,2);
    h{1}= X;
    for l = 1:nb_layers-1
        s{l} = W{l}*h{l} +b{l};
        h{l+1} = max(s{l},0);
    end
    s{nb_layers} = W{nb_layers}*h{nb_layers} +b{nb_layers};
    denom = sum(exp(s{nb_layers}),1);
    P = zeros(size(W{nb_layers},1),size(X,2));
    for i =1:size(s{nb_layers},2)
        P(:,i) = exp(s{nb_layers}(:,i))/denom(i);
    end
end