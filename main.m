rng(400);
GDparams = struct('n_batch',100,'eta_min',1e-5,'eta_max',1e-1,'n_s', 800, 'nb_cycles',2,'lambda',.01);

            
         

% Use all avaialable data
[data_batch_1X, data_batch_1Y, data_batch_1y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
[data_batch_2X, data_batch_2Y, data_batch_2y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
[data_batch_3X, data_batch_3Y, data_batch_3y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_3.mat');
[data_batch_4X, data_batch_4Y, data_batch_4y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_4.mat');
[data_batch_5X, data_batch_5Y, data_batch_5y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_5.mat');
[testX, testY, testy] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');
trainX = [data_batch_1X data_batch_2X data_batch_3X data_batch_4X data_batch_5X(:,1:5000)];
trainY = [data_batch_1Y data_batch_2Y data_batch_3Y data_batch_4Y data_batch_5Y(:,1:5000)];
trainy = [data_batch_1y;data_batch_2y;data_batch_3y;data_batch_4y;data_batch_5y(1:5000)];
validX = data_batch_5X(:,5000:10000);
validY = data_batch_5Y(:,5000:10000);
validy = data_batch_5y(5000:10000);

% preprocess data
trainX = PreProcess(trainX);
validX = PreProcess(validX);
testX = PreProcess(testX);
K = size(trainY,1);
d = size(trainX,1);
m = 50;
nb_layers=2;
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};
%{
whos W1
whos b1
whos W2
whos b2
%}
n = size(trainX,2);
GDparams.n_s = 2*floor(n / GDparams.n_batch);





fileID = fopen('Results_broad_lambda_search.txt','w');
fprintf(fileID,strcat(title,'\n'));
l_min=-5;
l_max=-1;
nb_uniform_tests = 7;
for i = 0:nb_uniform_tests
    l = l_min + i*(l_max-l_min)/nb_uniform_tests;
    GDparams.lambda = 10^l;
    [W,b] = init_params(K,d,m);
    title = strcat('nbatch=',string(GDparams.n_batch),',etamin=',string(GDparams.eta_min),...
                ',etamax=',string(GDparams.eta_max),',ns=',string(GDparams.n_s),...
                ',lambda=',string(GDparams.lambda),',nbcycles=',string(GDparams.nb_cycles));
    [Wstar, bstar] = MiniBatchGDWithPlots(trainX, trainY,trainy,validX, validY,validy, GDparams, W, b, title);
    final__test_acc = ComputeAccuracy(testX, testy, Wstar, bstar);
    [final__test_cost,final__test_loss] = ComputeCost(testX, testY, Wstar, bstar,GDparams.lambda);
    fprintf(fileID,' lamda=%f acc: %f loss: %f cost: %f \n',GDparams.lambda,final__test_acc,final__test_loss,final__test_cost);
end
fclose(fileID);


%{
% checking gradient

gradTestX = trainX(1:20, 1:5);
gradTestY = trainY(:, 1:5);
gradTestW1 = W1(:, 1:20);
gradTestW2 = W2;
gradTestW = {gradTestW1,gradTestW2};

eps = 1e-6;
[gradTestP,h,s] = EvaluateClassifier(gradTestX, gradTestW, b);
[grad_W, grad_b] = ComputeGradients(h, gradTestY, gradTestP, gradTestW, s, GDparams.lambda);
[ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(gradTestX, gradTestY, gradTestW, b, GDparams.lambda, 1e-6);
[ngrad_b_fast, ngrad_W_fast] = ComputeGradsNum(gradTestX, gradTestY, gradTestW, b, GDparams.lambda, 1e-6);
for l = 1:nb_layers
    assert(testSame(grad_W{l},ngrad_W_slow{l}, eps));
    assert(testSame(grad_b{l},ngrad_b_slow{l}, eps));

end

%{
Sanity check. Train 100 training samples for long time to get low test
error
%}
GDparams_test = struct('n_batch',10,'eta',0.001,'n_epochs',200, 'lambda',0);
trainX_test = trainX(:,1:100);
trainY_test = trainY(:,1:100);
trainy_test = trainy(1:100);
[W_test,b_test] = init_params(K,d,m);
[Wstar, bstar] = MiniBatchGDWithPlots(trainX_test, trainY_test, validX, validY, GDparams_test, W_test, b_test, title);
final_train_acc = ComputeAccuracy(trainX_test, trainy_test, Wstar, bstar)
%}
