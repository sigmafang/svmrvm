% Property 4.88 is used for calcuation of logistic sigmoid. 
% This function implements IRLS for calculating weights
% At the convergence of IRLS, the -ve hessian represents the inverse
% covariance matrix for the Gaussian approximation to the posterior
% Uses equations 7.112 and 7.113
% Refer to Michael Tipping's implementation for convergence criteria

function [w, covar] = updateWeightsAndCovariance(PHI,t,w,alpha)

maxIter = 25;
% Refernce : Michael Tipping's rvm for convergence
% This is the per-parameter gradient-norm threshold for termination
GRAD_STOP	= 1e-6;
% Limit to resolution of search step
LAMBDA_MIN	= 2^(-8);

[N d]	= size(PHI); 
M       = length(w);
A       = diag(alpha);
error	= zeros(maxIter,1); 
PHIw	= PHI*w;
y       = sigmoid(PHIw);
t       = logical(t);

% Compute initial value of log posterior (as an error)
% Equation 7.109 with split terms
errorTerm           = -(sum(log(y(t))) + sum(log(1-y(~t))))/N;
regularizationTerm	= (alpha'*(w.^2))/(2*N);
newError            =  errorTerm + regularizationTerm;

for i=1:maxIter
  
  bn	= y.*(1-y); % Elements of diagonal matrix B)
  e     = (t-y); 
  % Find gradient of posterior and Hessian matrix 
  g		= PHI'*e - alpha.*w; % Equation 7.110
  PHItemp	= PHI .* (bn * ones(1,d));
  Hessian	= (PHItemp'*PHI + A); % Equation 7.111
  error(i)   = newError;

  % Check for convergence
  if i>=2 && norm(g)/M<GRAD_STOP
      error    = error(1:i);
    break;
  end
  
  % Do Newton Raphson update and check see if error has decreased. 
  % If error increases, reduce the step and if error decreases, then
  % retain the weight calculated
  % Reference: Michael Tipping RVM implementation
  U         = chol(Hessian);
  delta_w	= U \ (U' \ g);
  lambda	= 1;
  while lambda > LAMBDA_MIN
    w_new	= w + lambda * delta_w; % Refer to equation 4.99
    PHIw	= PHI*w_new;
    y		= sigmoid(PHIw);

    % Compute new error 
    if any(y(t)==0) || any(y(~t)==1)
      newError	= inf; % Sigmoid cannot evaluate to 0 or 1
    else
      errorTerm             = -(sum(log(y(t))) + sum(log(1-y(~t))))/N;
      regularizationTerm	= (alpha'*(w_new.^2))/(2*N);
      newError              =  errorTerm + regularizationTerm;
    end
    if newError > error(i)
      % If error has increased, reduce lambda
      lambda	= lambda/2;
    else
      % If error has decreased, quit
      w         = w_new;
      lambda	= 0;
    end
  end
end
%
% Compute requisite values at mode
% 
covar      = inv(U);



