rng(400);
lambda = .1;
GDparams = struct('n_batch',100,'eta',0.001,'n_epochs',40);
title = 'lambda=.1, n epochs=40, n batch=100, eta=.001';



% Use all avaialable data
[data_batch_1X, data_batch_1Y, data_batch_1y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
[data_batch_2X, data_batch_2Y, data_batch_2y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
[data_batch_3X, data_batch_3Y, data_batch_3y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_3.mat');
[data_batch_4X, data_batch_4Y, data_batch_4y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_4.mat');
[data_batch_5X, data_batch_5Y, data_batch_5y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_5.mat');
[testX, testY, testy] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');
trainX = [data_batch_1X data_batch_2X data_batch_3X data_batch_4X data_batch_5X(:,1:9000)];
trainY = [data_batch_1Y data_batch_2Y data_batch_3Y data_batch_4Y data_batch_5Y(:,1:9000)];
trainy = [data_batch_1y;data_batch_2y;data_batch_3y;data_batch_4y;data_batch_5y(1:9000)];
validationX = data_batch_5X(:,9000:10000);
validationY = data_batch_5Y(:,9000:10000);
validationy = data_batch_5y(9000:10000);


trainX = PreProcess(trainX);
validX = PreProcess(validX);
testX = PreProcess(testX);

K = size(trainY,1);
d = size(trainX,1);
W = 0.01*randn(K,d);
b = 0.01*randn(K,1);

[Wstar, bstar] = MiniBatchGDWithPlots(trainX, trainY, validX, validY, GDparams, W, b, lambda, title);
finalAcc = ComputeAccuracy(testX, testy, Wstar, bstar)

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure
montage(s_im, 'Size', [5,5]);








