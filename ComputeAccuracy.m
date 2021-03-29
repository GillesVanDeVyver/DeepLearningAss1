function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~,I] = max(P);
    acc = sum(permute(I,[2,1])==y)/size(X,2);
end