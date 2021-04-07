rng(400);
lambda = 1;
GDparams = struct('n_batch',100,'eta',0.001,'n_epochs',40);
title = 'lambda=1, n epochs=40, n batch=100, eta=.001';

[trainX, trainY, trainy] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
[validX, validY, validy] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
[testX, testY, testy] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');

% preprocess data
trainX = PreProcess(trainX);
validX = PreProcess(validX);
testX = PreProcess(testX);
K = size(trainY,1);
d = size(trainX,1);
m = 50;
nb_layers=2;
[W,b] = init_params(K,d,m);
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

whos W1
whos b1
whos W2
whos b2

%{

% random initialisation

W = 0.01*randn(K,d);
b = 0.01*randn(K,1);

[Wstar, bstar] = MiniBatchGDWithPlots(trainX, trainY, validX, validY, GDparams, W, b, lambda, title);

% print final accuracy
finalAcc = ComputeAccuracy(testX, testy, Wstar, bstar)

% plot resulting images
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure
montage(s_im, 'Size', [5,5]);

%}

% checking gradient

gradTestX = trainX(1:20, 1);
gradTestY = trainY(:, 1);
gradTestW1 = W1(:, 1:20);
gradTestW2 = W2;
gradTestW = {gradTestW1,gradTestW2};

eps = 1e-6;
[gradTestP,h,s] = EvaluateClassifier(gradTestX, gradTestW, b);
[grad_W, grad_b] = ComputeGradients(h, gradTestY, gradTestP, gradTestW, s, lambda);
[ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(gradTestX, gradTestY, gradTestW, b, lambda, 1e-6);
[ngrad_b_fast, ngrad_W_fast] = ComputeGradsNum(gradTestX, gradTestY, gradTestW, b, lambda, 1e-6);
grad_W1 = grad_W{1};
for l = 1:nb_layers
    assert(testSame(grad_W{l},ngrad_W_slow{l}, eps));
    assert(testSame(grad_b{l},ngrad_b_slow{l}, eps));

end


