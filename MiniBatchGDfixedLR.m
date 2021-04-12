function [Wstar, bstar] = MiniBatchGDfixedLR(X, Y, GDparamsfixedLR, W, b)
    n_batch = GDparamsfixedLR.n_batch;
    eta = GDparamsfixedLR.eta;
    n_epochs = GDparamsfixedLR.n_epochs;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    nb_layers = size(W,2);
    for i=1:n_epochs
        shuffleInds = randperm(n);
        Xshuffle = X(:, shuffleInds);
        Yshuffle = Y(:, shuffleInds);
        for j=1:n/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xshuffle(:, j_start:j_end);
            Ybatch = Yshuffle(:, j_start:j_end);
            [Pbacth,h,s] = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(h, Ybatch, Pbacth, Wstar, s, GDparamsfixedLR.lambda);
            for l = 1:nb_layers
                Wstar{l} = Wstar{l} - eta*grad_W{l};
                bstar{l} = bstar{l} - eta*grad_b{l};
            end

        end
    end

end


