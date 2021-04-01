function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    rng(400);
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar=W;
    bstar=b;
    %init_cost = ComputeCost(X, Y, Wstar, bstar,lambda)
    n = size(X,2);
    for i=1:n_epochs
        shuffleInds = randperm(n);
        Xshuffle = X(:, shuffleInds);
        Yshuffle = Y(:, shuffleInds);
        for j=1:n/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = Xshuffle(:, j_start:j_end);
            Ybatch = Yshuffle(:, j_start:j_end);
            Pbacth = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, Pbacth, Wstar, lambda);
            Wstar = Wstar - eta*grad_W;
            bstar = bstar - eta*grad_b;
        end
        %i
        %cost = ComputeCost(X, Y, Wstar, bstar,lambda)
    end
end


