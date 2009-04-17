clc;
clear;
close all;

load 'F:\Project2\GunnExample\iris3v12.mat';
% groups = ismember(Y,1);
indices = crossvalind('Kfold',Y,10);
cp = classperf(Y);
for i = 1:10
    test = (indices == i); train = ~test;
    
    C = 50;
    trnX = X(train,:);
    trnY = Y(train,:);
    tstX = X(test,:);
    tstY = Y(test,:);
    
    [nsv alpha bias] = svc(trnX,trnY,'rbf',C);
    predictedY = svcoutput(trnX,trnY,tstX,'rbf',alpha,bias,0);
    round = i
    err = svcerror(trnX,trnY,tstX,tstY,'rbf',alpha,bias)
    figure
    svcplot(trnX,trnY,'rbf',alpha,bias,0);
end




