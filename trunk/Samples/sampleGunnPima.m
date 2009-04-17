clc;
clear;
close all;
% Pima attributes
%    1. Number of times pregnant
%    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
%    3. Diastolic blood pressure (mm Hg)
%    4. Triceps skin fold thickness (mm)
%    5. 2-Hour serum insulin (mu U/ml)
%    6. Body mass index (weight in kg/(height in m)^2)
%    7. Diabetes pedigree function
%    8. Age (years)

load 'F:\Project2\svmrvm\pima.txt';
data = pima(:,1:8);

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

% figure
% scatter3(scores(:,1),scores(:,2),scores(:,3), 30 ,Y, 'filled');
% xlabel('# times pregnant(PC1)');
% ylabel('Plama Glucose conc(PC2)');
% zlabel('Diastolic Blood Pressure(PC3)')
% title('3D principle components');
% hold on;
% figure
% bar(percent_explained)
% xlabel('Principal Component')
% ylabel('Variance Explained (%)')
% title('Principle Components');

% Try to use SVM and see error in 10 fold cross validation
% clear;
% load 'F:\Project2\svmrvm\pima1.mat';
% groups = ismember(Y,1);
% Recompile qp.dll
cd 'C:\Program Files\MATLAB\R2007a\toolbox\svm\Optimiser'
mex qp.c pr_loqo.c
!copy "C:\Program Files\MATLAB\R2007a\toolbox\svm\Optimiser\qp.mexw32" "C:\Program Files\MATLAB\R2007a\toolbox\svm"

indices = crossvalind('Kfold',Y,10);
C = 5000;
global p1;
p1 = 0.1;
global p2
p2 = 0;
global sep;
sep = 1;

cp = classperf(Y);
for i = 1:1
    test = (indices == i); train = ~test;
    trnX = scores(train,:);
    trnY = Y(train,:);
    tstX = scores(test,:);
    tstY = Y(test,:);
    
    [nsv alpha bias] = svc(trnX,trnY,'rbf',C);
    predictedY = svcoutput(trnX,trnY,tstX,'rbf',alpha,bias,0);
    round = i
    err = svcerror(trnX,trnY,tstX,tstY,'rbf',alpha,bias)
    
%     figure
%     svcplot(trnX(:,1:2),trnY,'rbf',alpha,bias,0);
    
    figure
    svcplot(scores(:,1:2),trnY,'rbf',alpha,bias,0);
end



