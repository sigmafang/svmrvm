clc;
clear;
close all;
fprintf('RVM for classification\n');

% Load the training data (the data has been split into training and test)
load 'pimaTrain8.mat';
X       = trnX;
t	= trnY;
[N m] = size(trnX);

% Plot the training data
figure
plot(X(t==0,1),X(t==0,2),'.','MarkerSize',15,'Color','r');
hold on
plot(X(t==1,1),X(t==1,2),'.','MarkerSize',15,'Color','g');


% Start with an initial guess of hyper parameters
alpha	= (1/N)^2;

% Create PHI matrix using RBF kernel and augmeent 1
PHI     = createPhiMat(X,X);
PHI     = [PHI ones(N,1)];

[weights, alpha, gamma, used] = ...
            getHyperParams(alpha, t, PHI);


% strip off bias for later convinience
bias	= 0;
indexBias	= find(used==N+1);
if ~isempty(indexBias)
    bias		= weights(indexBias);
    used(indexBias)	= [];
    weights(indexBias)	= [];
end

% Load the test data
load 'pimaTest8.mat';
Xtest	= tstX;
ttest	= tstY;
[Nt mt] = size(tstX);

%
% Compute RVM over test data and calculate error
% 
PHI     =  createPhiMat(Xtest,X(used,:));
y_rvm	= PHI*weights + bias;
errs	= sum(y_rvm(ttest==0)>0) + sum(y_rvm(ttest==1)<=0);
fprintf('RVM CLASSIFICATION test error: %.2f%%\n', errs/Nt*100);

box	= 1.1*[min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))];
gsteps		= 50;
range1		= box(1):(box(2)-box(1))/(gsteps-1):box(2);
range2		= box(3):(box(4)-box(3))/(gsteps-1):box(4);
[grid1 grid2]	= meshgrid(range1,range2);
Xgrid		= [grid1(:) grid2(:)];
%
% Evaluate RVM
% 
PHI		= createPhiMat(Xgrid,X(used,1:2));
y_grid		= PHI*weights + bias;
% apply sigmoid for probabilities
p_grid		= 1./(1+exp(-y_grid)); 
%
% Show decision boundary (p=0.5) and illustrate p=0.25 and 0.75
% 
[c,h05]		= ...
    contour(range1,range2,reshape(p_grid,size(grid1)),[0.5],'-');
[c,h075]	= ...
    contour(range1,range2,reshape(p_grid,size(grid1)),[0.25 0.75],'--');
set(h05, 'Color','black','LineWidth',3);
set(h075,'Color','blue','LineWidth',2);
%
% Show relevance vectors
% 
h_rv	= plot(X(used,1),X(used,2),'o','LineWidth',2,'MarkerSize',10,...
	       'Color','green');
%
% legend([h_c1 h_c2 h05 h075(1) h_rv],...
%        'Class 1','Class 2','Decision boundary','p=0.25/0.75','RVs',...
%        'Location','NorthWest')
% %
% hold off
% title('RVM Classification of Ripley''s synthetic data','FontSize',14)
% 



