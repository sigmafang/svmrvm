clc;
clear;
close all;
cd 'C:\Program Files\MATLAB\R2007a\toolbox\svm\Optimiser'
mex qp.c pr_loqo.c
!copy "C:\Program Files\MATLAB\R2007a\toolbox\svm\Optimiser\qp.mexw32" "C:\Program Files\MATLAB\R2007a\toolbox\svm"
%cd 'F:\Project2\svmrvm\'
uiclass
