function [W,b] = init_params(K,d,m)
    W1 = 1/sqrt(d)*randn(m,d);
    W2 = 1/sqrt(m)*randn(K,m);
    b1 = zeros(m,1);
    b2 = zeros(K,1);
    W = {W1,W2};
    b = {b1,b2};
end