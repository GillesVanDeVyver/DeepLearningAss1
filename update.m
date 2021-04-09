function [Wstar,bstar,j,X,Y,costTrain,costValid,eval_step] = update(eta,n,X,Y,XValid, YValid,Wstar,bstar,GDparams,nb_layers,j,t,eval_interval,eval_step,costTrain,costValid)

    j_start = (j-1)*GDparams.n_batch + 1;
    j_end = j*GDparams.n_batch;
    Xbatch = X(:, j_start:j_end);
    Ybatch = Y(:, j_start:j_end);
    [Pbacth,h,s] = EvaluateClassifier(Xbatch, Wstar, bstar);
    [grad_W, grad_b] = ComputeGradients(h, Ybatch, Pbacth, Wstar, s, GDparams.lambda);
    for l = 1:nb_layers
        Wstar{l} = Wstar{l} - eta*grad_W{l};
        bstar{l} = bstar{l} - eta*grad_b{l};
    end
    if j < n/GDparams.n_batch
        j = j+1;
    else
        j=1;
        shuffleInds = randperm(n);
        X = X(:, shuffleInds);
        Y = Y(:, shuffleInds);
    end
    if mod(t,eval_interval) == 0
        costTrain(eval_step) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
        costValid(eval_step) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
        eval_step = eval_step+1;
        t
    end
end