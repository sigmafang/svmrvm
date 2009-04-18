function PHI = createPhiMat(X,Y)
[N1 d]		= size(X);
[N2 d]		= size(Y);
sigma       = 0.5;
sigmasq     = sigma*sigma;
constant    = -(0.5/sigmasq);
dist    	= sum(X.^2,2)*ones(1,N2) + ones(N1,1)*sum(Y.^2,2)' - 2*(X*Y');
PHI         = exp(constant*dist);
end