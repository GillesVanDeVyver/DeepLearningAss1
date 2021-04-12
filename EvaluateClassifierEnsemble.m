function P = EvaluateClassifierEnsemble(X, Wenseble, bensemble)
    nb_layers = size(Wenseble{1},2);
    P_sum=zeros(size(Wenseble{1}{nb_layers},1),size(X,2));
    for j=1:size(Wenseble)
        W=Wenseble{j};
        b=bensemble{j};
        [Pj,~,~] = EvaluateClassifier(X, W, b);
        P_sum=P_sum+Pj;
    end
    P = P_sum/size(Wenseble,1);
end