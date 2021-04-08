function [grad_W, grad_b] = ComputeGradients(h, Y, P, W, s, lambda)
    n = size(h{1},2); %h{1} = input data
    g = -(Y-P).';
    nb_layers = size(W,2);
    for l = nb_layers:-1:2
        grad_W{l} = 1/n*g.'*h{l}.' + 2*lambda*W{l};
        grad_b{l} = 1/n*g.'*ones(n,1);
        g = g*W{l};
        g = g.*((s{l-1}>0).');
    end
    grad_W{1} = 1/n*g.'*h{1}.' + 2*lambda*W{1};
    grad_b{1} = 1/n*g.'*ones(n,1);

end