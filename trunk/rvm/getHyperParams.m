function [weights, alpha, gamma, reqdIndices] = getHyperParams(alpha, t, PHI)

    minLnAlphaChange	= 1e-3;
    maxAlpha            = 1e9;
    maxIts              = 1000;

    % Set up parameters and hyperparameters
    [N,M]       = size(PHI); 
    weights           = zeros(M,1); 
    alpha       = alpha*ones(M,1);
    gamma       = ones(M,1);
    isLastIter	= 0;

    for i   =  1:maxIts
      % Remove large alpha values, set corresponsing weights to zero and
      % take only corresponding basis functions as well
      % Refer to Michael tipping's algorithm for pruning criteria
      reqdIndices       = (alpha < maxAlpha);
      alpha_used        = alpha(reqdIndices);
      M                 = sum(reqdIndices);
      weights(~reqdIndices)	= 0;
      PHI_used          = PHI(:,reqdIndices);

      [weights(reqdIndices) Ui likelihood] = ...
        updateWeightsAndCovariance(PHI_used,t,weights(reqdIndices),alpha_used);

      % Need determinant and diagonal values of 
      % posterior weight covariance matrix (SIGMA in paper)
      logdetH	= -2*sum(log(diag(Ui)));
      diagSig	= sum(Ui.^2,2);
      gamma		= 1 - alpha_used.*diagSig;

      % Compute marginal likelihood (approximation for classification case)
      marginal	= likelihood - 0.5*(logdetH - sum(log(alpha_used)) + ...
                       (weights(reqdIndices).^2)'*alpha_used);

      % print out how many non zero params are retained
      fprintf('iter = %d number of non zero params = %d\n',...
         i, sum(reqdIndices));

      if ~isLastIter
        logAlpha		= log(alpha(reqdIndices));
        % update alpha using Eqn 7.116
        alpha(reqdIndices)	= gamma ./ weights(reqdIndices).^2;
       
        % Find difference in the lod alpha values.
        % If difference is less than min diference set, break from loop
        reqdAlpha       = alpha(reqdIndices);
        logAlphaDiff    = abs(logAlpha(reqdAlpha~=0) - log(reqdAlpha(reqdAlpha~=0)));
        maxDAlpha       = max(logAlphaDiff);
        if maxDAlpha < minLnAlphaChange
              isLastIter	= 1; %The change in alpha vreqdAlphaes is smaller than that is required
              fprintf('Terminating as change in log alpha stagnates...\n');
        end
      else
        % Break if isLastIter = true
        break;	
      end
    end

    % Copy to return values
    weights = weights(reqdIndices);
    reqdIndices	= find(reqdIndices);

    if ~isLastIter
      fprintf('Terminating as max number of iterations reached..\n');
    end
    
    fprintf('Number of non zero parameters is %d\n', length(weights));
end