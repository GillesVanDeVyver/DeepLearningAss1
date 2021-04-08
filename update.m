function [Wstar,bstar,j] = update(eta,n,X,Y,Wstar,bstar,GDparams,nb_layers,j)
    shuffleInds = randperm(n);
    Xshuffle = X(:, shuffleInds);
    Yshuffle = Y(:, shuffleInds);
    %for j=1:n/GDparams.n_batch
    j_start = (j-1)*GDparams.n_batch + 1;
    j_end = j*GDparams.n_batch;
    Xbatch = Xshuffle(:, j_start:j_end);
    Ybatch = Yshuffle(:, j_start:j_end);
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
    end

    %end
end