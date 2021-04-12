function LRrangeTest(trainX, trainY,trainy, GDparams,K,d)

    GDparamsfixedLR = struct('n_batch',215,'eta',1e-5,'n_epochs', 8,'lambda',0.0025119);

    exp_eta_min = -5;
    exp_eta_max = 0;
    nb_steps = 20;
    train_accuracies = zeros(nb_steps+1,1);
    x_axis = zeros(nb_steps+1,1);
    i=1;
    for step =0:nb_steps
        exp_eta = exp_eta_min + (exp_eta_max-exp_eta_min)*step/nb_steps;
        GDparamsfixedLR.eta=10^exp_eta;
        [W,b] = init_params(K,d,GDparams.m);
        [Wstar, bstar] = MiniBatchGDfixedLR(trainX, trainY, GDparamsfixedLR, W, b);
        train_accuracies(i) = ComputeAccuracy(trainX, trainy, Wstar, bstar);
        x_axis(i) = GDparamsfixedLR.eta;
        i=i+1;
    end
    
    semilogx(x_axis,train_accuracies)
    xlabel('eta') 
    ylabel('accuracy')
    legend({'training accuracy'})
    plotTitle = 'LR range test';
    title(plotTitle)
    ylim([0 max(train_accuracies)+1])
    xlim([0 inf]);
    saveas(1,strcat(plotTitle,'.png'))
end