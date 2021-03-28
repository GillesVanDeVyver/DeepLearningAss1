

% Read in and store the training, validation and test data.
[trainX, trainY, trainy] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
[validX, validY, validy] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
[testX, testY, testy] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');

%{
Compute the mean and standard deviation vector for the
training data and then normalize the training, validation and test data
w.r.t. these mean and standard deviation vectors.
%}
trainX = PreProcess(trainX);
validX = PreProcess(validX);
testX = PreProcess(testX);

%{
After reading in and pre-processing the data, you can
initialize the parameters of the model W and b as you now know what
size they should be. W has size K×d and b is K×1. Initialize each
entry to have Gaussian random values with zero mean and standard
deviation .01. You should use the Matlab function randn to create
this data.
%}
K = size(trainY,1);
d = size(trainX,1);
W = 0.01*randn(K,d);
b = 0.01*randn(K,1);


%P = EvaluateClassifier(trainX(:, 1:100), W, b)
%J = ComputeCost(trainX(:, 1:100), trainY(:, 1:100), W, b, 0.01)
acc = ComputeAccuracy(trainX(:, 1:1000), trainy(1:1000), W, b)

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = double(permute(A.data, [2,1]));
    y = double(A.labels+1);
    Y = double(permute(y==1:10,[2,1]));
end

function X = PreProcess(X)
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end

% P = Kxn
% X = dxn
% Y = Kxn one hot
% y = 1xn
function P = EvaluateClassifier(X, W, b)
    s = W*X +b;
    denom = sum(exp(s),1);
    P = zeros(size(W,1),size(X,2));
    for i =1:size(s,2)
        P(:,i) = exp(s(:,i))/denom(i);
    end
end

function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    lcross = -log(dot(Y,P));
    J = sum(lcross)/size(X,2);
    W_squared = W.^2;
    J = J + lambda*sum(W_squared(:));
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~,I] = max(P);
    acc = sum(permute(I,[2,1])==y)/size(X,2);
end



