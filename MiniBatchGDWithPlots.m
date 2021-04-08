function [Wstar, bstar] = MiniBatchGDWithPlots(X, Y, XValid, YValid, GDparams, W, b, plotTitle)
    rng(400);
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    costTrain = zeros(n_epochs+1,1);
    costValid = zeros(n_epochs+1,1);
    costTrain(1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
    costValid(1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
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
            [grad_W, grad_b] = ComputeGradients(h, Ybatch, Pbacth, Wstar, s, GDparams.lambda);
            for l = 1:nb_layers
                Wstar{l} = Wstar{l} - eta*grad_W{l};
                bstar{l} = bstar{l} - eta*grad_b{l};
            end

        end
        costTrain(i+1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
        costValid(i+1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
    end
    epochInds = 0:n_epochs;
    figure
    plot(epochInds,costTrain,epochInds,costValid)
    xlabel('epoch') 
    ylabel('loss')
    legend({'training loss','validation loss'},'Location','northeast')
    title(plotTitle)
    axis tight
end


