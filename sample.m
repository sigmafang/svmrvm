clc;
clear;
close all;

load ('pima.txt');
pdata = pima(:,1:8);
class = pima(:,9);
[coefs,scores,variances,t2] = princomp(pdata);
percent_explained = 100*variances/sum(variances);

%gscatter(scores(:,1),scores(:,2),class,'rgb')
gscatter(pima(:,1),pima(:,2),class,'rgb')
xlabel('# of times pregnant');
ylabel('% Plasma glucose concentration');
title('2-D principle components');

