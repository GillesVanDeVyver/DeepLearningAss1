function X = PreProcess(X)
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end