function [Wstar, bstar] = MiniBatchGDWithPlots(X, Y, XValid, YValid, GDparams, W, b, lambda, plotTitle)
    rng(400);
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    costTrain = zeros(n_epochs+1);
    costValid = zeros(n_epochs+1);
    costTrain(1) = ComputeCost(X, Y, Wstar, bstar,lambda);
    costValid(1) = ComputeCost(XValid, YValid, Wstar, bstar,lambda);
    for i=1:n_epochs
        shuffleInds = randperm(n);
        Xshuffle = X(:, shuffleInds);
        Yshuffle = Y(:, shuffleInds);
        for j=1:n/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xshuffle(:, j_start:j_end);
            Ybatch = Yshuffle(:, j_start:j_end);
            Pbacth = EvaluateClassifier(Xbatch, Wstar, bstar);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, Pbacth, Wstar, lambda);
            Wstar = Wstar - eta*grad_W;
            bstar = bstar - eta*grad_b;
        end
        %i
        costTrain(i+1) = ComputeCost(X, Y, Wstar, bstar,lambda);
        costValid(i+1) = ComputeCost(XValid, YValid, Wstar, bstar,lambda);
    end
    epochInds = 0:n_epochs;
    figure
    plot(epochInds,costTrain,epochInds,costValid)
    %ylim([1.5 2.5])
    xlabel('epoch') 
    ylabel('loss')
    legend({'training loss','validation loss'},'Location','northeast')
    title(plotTitle)
    axis tight


end

