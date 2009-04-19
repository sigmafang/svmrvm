function PHI = createPhiMat(X,Y)
N1          = size(X,1);
N2          = size(Y,1);
sigma       = 0.5;
sigmasq     = sigma*sigma;
constant    = -(1/sigmasq);
distsq    	= sum(X.^2,2)*ones(1,N2) + ones(N1,1)*sum(Y.^2,2)' - 2*(X*Y');
PHI         = exp(constant*distsq);
end