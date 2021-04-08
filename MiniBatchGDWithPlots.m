function [Wstar, bstar] = MiniBatchGDWithPlots(X, Y, XValid, YValid, GDparams, W, b, plotTitle)
    rng(400);
    n_s = GDparams.n_s;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    total_nb_steps = GDparams.nb_cycles*n_s*2;
    costTrain = zeros(total_nb_steps+1,1);
    costValid = zeros(total_nb_steps+1,1);
    costTrain(1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
    costValid(1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
    nb_layers = size(W,2);
    t=0;
    for l=0:GDparams.nb_cycles-1
        t
        while t<(2*l+1)*n_s
            eta = GDparams.eta_min+(t-2*l*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar] = update(eta,n,X,Y,Wstar,bstar,GDparams,nb_layers);
            costTrain(t+1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
            costValid(t+1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
            t=t+1;
        end
        t
        while t<2*(l+1)*n_s
            eta= GDparams.eta_max-(t-(2*l+1)*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar] = update(eta,n,X,Y,Wstar,bstar,GDparams,nb_layers);
            costTrain(t+1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
            costValid(t+1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
            t=t+1;
        end
    end
    x_axis = 0:total_nb_steps;
    figure
    whos x_axis
    whos costTrain
    whos costValid
    plot(x_axis,costTrain,x_axis,costValid)
    xlabel('update step') 
    ylabel('loss')
    legend({'training loss','validation loss'},'Location','northeast')
    title(plotTitle)
    axis tight
    print -depsc cost_exercise_4
end


