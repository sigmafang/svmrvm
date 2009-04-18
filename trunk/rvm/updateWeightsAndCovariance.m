function [w, Ui, lMode] = updateWeightsAndCovariance(PHI,t,w,alpha)

its = 25;
% This is the per-parameter gradient-norm threshold for termination
GRAD_STOP	= 1e-6;
% Limit to resolution of search step
LAMBDA_MIN	= 2^(-8);

[N d]	= size(PHI);
M 	= length(w);
A	= diag(alpha);
errs	= zeros(its,1);
PHIw	= PHI*w;
y	= sigmoid(PHIw);
t	= logical(t);

% Compute initial value of log posterior (as an error)
% 
data_term	= -(sum(log(y(t))) + sum(log(1-y(~t))))/N;
regulariser	= (alpha'*(w.^2))/(2*N);
err_new		=  data_term + regulariser;

for i=1:its
  %
  yvar	= y.*(1-y);
  PHIV	= PHI .* (yvar * ones(1,d));
  e	= (t-y);
  %
  % Compute gradient vector and Hessian matrix
  % 
  g		= PHI'*e - alpha.*w;
  Hessian	= (PHIV'*PHI + A);
  
  errs(i)   = err_new;
  %
  % See if converged
  % 
  if i>=2 & norm(g)/M<GRAD_STOP
      errs    = errs(1:i);
%     fprintf('Posterior mode converged');
    break;
  end
  %
  % Take "Newton step" and check for reduction in error
  % 
  U		= chol(Hessian);
  delta_w	= U \ (U' \ g);
  lambda	= 1;
  while lambda>LAMBDA_MIN
    w_new	= w + lambda*delta_w;
    PHIw	= PHI*w_new;
    y		= sigmoid(PHIw);
    %
    % Compute new error 
    % 
    if any(y(t)==0) | any(y(~t)==1)
      err_new	= inf;
    else
      data_term		= -(sum(log(y(t))) + sum(log(1-y(~t))))/N;
      regulariser	= (alpha'*(w_new.^2))/(2*N);
      err_new		=  data_term + regulariser;
    end
    if err_new>errs(i)
      %
      % If error has increased, reduce the step
      % 
      lambda	= lambda/2;
      
    else
      %
      % Error has gone down: accept the step
      % 
      
      w		= w_new;
      lambda	= 0;
    end
  end
  %
  % If we're here with non-zero lambda, then we couldn't take a small
  % enough downhill step: converged close to minimum
  % 
%   if lambda
%     fprintf('Posterior mode stopping as back off limit is reached');
%     break;
%   end
end
%
% Compute requisite values at mode
% 
Ui	= inv(U);
lMode	= -N*data_term;

%%
%% Support function: sigmoid
%%
function y = sigmoid(x)
y = 1./(1+exp(-x));