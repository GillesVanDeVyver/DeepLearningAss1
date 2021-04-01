function [grad_W, grad_b] = ComputeGradientsSVM(X, Y, b, W, lambda)
    n = size(X,2);
    s = W*X+b;
    sy = sum(Y.*s, 1);
    Ynot = double(~Y);
    loss = Ynot.*(s-sy+1);
    %whos loss
    %loss(:,1)
    miss_classification = double(loss>0);
    %whos miss_classification
    %miss_classification(:,1)
    %multi_class_loss = max(loss,[],1);
    %whos multi_class_loss
    miss_sum= sum(miss_classification);
    %whos miss_sum
    %miss_sum(1)
    grady = -miss_sum.*Y;
    %whos grady
    %grady(:,1)
    
    
    
    %G_batch = -(Y-P);
    %whos G_batch
    grad_W = 1/n*(miss_classification+grady)*X.' +2*lambda*W;
    grad_b = 1/n*(miss_classification+grady)*ones(n,1);
    %whos grad_W
    %whos grad_b
    
end