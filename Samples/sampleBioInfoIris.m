clc;
clear;
close all;

% View 2 principle components of PIMA data
% load ('pima.txt');
% pdata = pima(:,1:8);
% class = pima(:,9);
% [coefs,scores,variances,t2] = princomp(pdata);
% percent_explained = 100*variances/sum(variances);
% 
% %gscatter(scores(:,1),scores(:,2),class,'rgb')
% gscatter(pima(:,1),pima(:,2),class,'rgb')
% xlabel('# of times pregnant');
% ylabel('% Plasma glucose concentration');
% title('2-D principle components');

% Try Gunn's sample iris data with matlab's bioinformatics toolbox svm impl
load 'gunnExample\iris3v12.mat'
data = [X(:,1), X(:,2)];
groups = ismember(Y,1);
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);
svmStruct = svmtrain(data(train,:),groups(train),'Kernel_Function', 'rbf', 'RBF_Sigma', 0.5, 'showplot',true)
title(sprintf('Kernel Function: %s',...
              func2str(svmStruct.KernelFunction)),...
              'interpreter','none');
classes = svmclassify(svmStruct,data(test,:),'showplot',true);
classperf(cp,classes,test);
cp.CorrectRate
figure
svmStruct = svmtrain(data(train,:),groups(train),...
                     'Kernel_Function', 'rbf', 'RBF_Sigma', 0.5,'showplot',true,'boxconstraint',1e6);
classes = svmclassify(svmStruct,data(test,:),'showplot',true);
classperf(cp,classes,test);
cp.CorrectRate

                 
