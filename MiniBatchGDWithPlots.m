function [Wstar, bstar] = MiniBatchGDWithPlots(X, Y,y, XValid, YValid, yvalid, GDparams, W, b, plotTitle,plot)
    rng(400);
    n_s = GDparams.n_s;
    Wstar=W;
    bstar=b;
    n = size(X,2);
    eval_per_cycle = 10;
    nb_eval_points = GDparams.nb_cycles*eval_per_cycle;
    eval_interval = floor((2*n_s)/eval_per_cycle);
    if plot
        cost_train = zeros(nb_eval_points+1,1);
        cost_valid = zeros(nb_eval_points+1,1);
        loss_train = zeros(nb_eval_points+1,1);
        loss_valid = zeros(nb_eval_points+1,1);
        acc_train = zeros(nb_eval_points+1,1);
        acc_valid = zeros(nb_eval_points+1,1);
        plot_info = {cost_train,cost_valid,loss_train,loss_valid,acc_train,acc_valid};
    else
       plot_info=0; 
    end
    nb_layers = size(W,2);
    t=0;
    eval_step = 1;
    batch_offset = 1;
    for l=0:GDparams.nb_cycles-1
        while t<(2*l+1)*n_s
            eta = GDparams.eta_min+(t-2*l*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar,batch_offset,X,Y,plot_info,eval_step,t] = update(eta,n,X,Y,y,XValid, YValid,yvalid,Wstar,bstar,GDparams,nb_layers,batch_offset,t,eval_interval,eval_step,plot_info,plot);
        end
        while t<2*(l+1)*n_s
            eta= GDparams.eta_max-(t-(2*l+1)*n_s)/n_s*(GDparams.eta_max-GDparams.eta_min);
            [Wstar,bstar,batch_offset,X,Y,plot_info,eval_step,t] = update(eta,n,X,Y,y,XValid, YValid,yvalid,Wstar,bstar,GDparams,nb_layers,batch_offset,t,eval_interval,eval_step,plot_info,plot);
        end
    end
    if plot
        [plot_info{1}(nb_eval_points+1),plot_info{3}(nb_eval_points+1)] = ComputeCost(X, Y, Wstar, bstar,GDparams.lambda);
        [plot_info{2}(nb_eval_points+1),plot_info{4}(nb_eval_points+1)] = ComputeCost(XValid, YValid, Wstar, bstar,GDparams.lambda);
        plot_info{5}(nb_eval_points+1)= ComputeAccuracy(X, y, Wstar, bstar);
        plot_info{6}(nb_eval_points+1)= ComputeAccuracy(XValid, yvalid, Wstar, bstar);
        x_axis = 0:nb_eval_points;
        x_axis = x_axis*eval_interval;
        figure('Renderer', 'painters', 'Position', [10 10 1500 300])
        tiledlayout(1,3)
        nexttile
        plot(x_axis,plot_info{1},x_axis,plot_info{2})
        ylim([0 max(plot_info{2})+1])
        xlabel('update step') 
        ylabel('cost')
        legend({'training cost','validation cost'},'Location','northeast')
        nexttile
        plot(x_axis,plot_info{3},x_axis,plot_info{4})
        ylim([0 max(plot_info{4})+1])
        xlabel('update step') 
        ylabel('loss')
        legend({'training loss','validation loss'},'Location','northeast')
        nexttile
        plot(x_axis,plot_info{5},x_axis,plot_info{6})
        ylim([0 1])
        xlabel('update step') 
        ylabel('accuracy')
        legend({'training accuracy','validation accuracy'},'Location','northeast')
        sgtitle(plotTitle) 
        saveas(1,strcat(strrep(plotTitle, '.', ','),'.png'))
    end
end


