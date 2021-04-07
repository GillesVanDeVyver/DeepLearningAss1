function [W1,b1,W2,b2] = init_params(K,d,m)
    W1 = num2cell(1/sqrt(d)*randn(m,d));
    W2 = num2cell(1/sqrt(m)*randn(K,m));
    b1 = num2cell(zeros(m,1));
    b2 = num2cell(zeros(K,1));
end