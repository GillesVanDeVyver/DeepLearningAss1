function acc = ComputeAccuracyEnsemble(X, y, W_enseble, b_ensemble)
    P = EvaluateClassifierEnsemble(X, W_enseble, b_ensemble);
    [~,I] = max(P);
    acc = sum(permute(I,[2,1])==y)/size(X,2);
end