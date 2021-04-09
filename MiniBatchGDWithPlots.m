function [Wstar, bstar] = MiniBatchGDWithPlots(X, Y, XValid, YValid, GDparams, W, b, plotTitle)
    rng(400);
    n_s = GDparams.n_s;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    eval_per_cycle = 10;
    nb_eval_points = GDparams.nb_cycles*eval_per_cycle;
    eval_interval = floor((2*n_s)/eval_per_cycle);
    costTrain = zeros(nb_eval_points+1,1);
    costValid = zeros(nb_eval_points+1,1);
    costTrain(1) = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
    costValid(1) = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
    nb_layers = size(W,2);
    t=0;
    eval_step = 2;
    batch_offset = 1;
    for l=0:GDparams.nb_cycles-1
        while t<(2*l+1)*n_s
            eta = GDparams.eta_min+(t-2*l*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar,batch_offset,X,Y,costTrain,costValid,eval_step] = update(eta,n,X,Y,XValid, YValid,Wstar,bstar,GDparams,nb_layers,batch_offset,t,eval_interval,eval_step,costTrain,costValid);
            t=t+1;
        end
        while t<2*(l+1)*n_s
            eta= GDparams.eta_max-(t-(2*l+1)*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar,batch_offset,X,Y,costTrain,costValid,eval_step] = update(eta,n,X,Y,XValid, YValid,Wstar,bstar,GDparams,nb_layers,batch_offset,t,eval_interval,eval_step,costTrain,costValid);
            t=t+1;

            
        end
    end
    x_axis = 0:nb_eval_points;
    x_axis = x_axis*eval_interval;
    figure
    plot(x_axis,costTrain,x_axis,costValid)
    ylim([0 inf])
    xlabel('update step') 
    ylabel('loss')
    legend({'training loss','validation loss'},'Location','northeast')
    title(plotTitle)
    print -depsc cost_exercise_4
end


