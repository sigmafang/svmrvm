clc;

load 'pima.txt';
% Impute the missing values in the data using knn
% zeroCount = sum(data == 0);
X = pima(:,1);
missingX2_8 = pima(:,2:8);
missingX2_8 = missingX2_8';

temp = (missingX2_8 == 0);
missingX2_8(temp) = NaN;
X2_8 = knnimpute(missingX2_8, 10);
X2_8 = X2_8';

% X holds the final attribute values for the dataset
X = [X X2_8];
Y = pima(:,9);

% Do PCA on the data and display 3D
p = X;
stdr = std(p);
[m, n] = size(p);
sr = p./repmat(stdr,m,1);
[coefs,scores,variances,t2] = princomp(sr);
percent_explained = 100*variances/sum(variances);
indices = crossvalind('Kfold',Y,10);

cp = classperf(Y);
for i = 1:1
    test = (indices == i); train = ~test;
    trnX = scores(train,:);
    trnY = Y(train,:);
    tstX = scores(test,:);
    tstY = Y(test,:);
end

% tstX = tstX(:,1:2);
% trnX = trnX(:,1:2);
save pimaTest8.mat tstX tstY;
save pimaTrain8.mat trnX trnY;