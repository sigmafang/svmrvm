function [weights, alpha, gamma, used] = getHyperParams(alpha, t, PHI)

    MIN_DELTA_LOGALPHA	= 1e-3;
    ALPHA_MAX           = 1e9;
    maxIts              = 1000;

    % Set up parameters and hyperparameters
    [N,M]	= size(PHI);
    w       = zeros(M,1); 
    alpha	= alpha*ones(M,1);
    gamma	= ones(M,1);
    LAST_IT	= 0;
    %
    % The main loop
    % 
    for i=1:maxIts
 
      % Prune based on large values of alpha
      useful        = (alpha<ALPHA_MAX);
      alpha_used	= alpha(useful);
      M             = sum(useful);
      % Prune weights and basis
      w(~useful)	= 0;
      PHI_used      = PHI(:,useful);


    [w(useful) Ui dataLikely] = ...
        updateWeightsAndCovariance(PHI_used,t,w(useful),alpha_used);

      % Need determinant and diagonal values of 
      % posterior weight covariance matrix (SIGMA in paper)
      logdetH	= -2*sum(log(diag(Ui)));
      diagSig	= sum(Ui.^2,2);
      %
      % Well-determinedness parameters (gamma)
      % 
      gamma		= 1 - alpha_used.*diagSig;

      %
      % Compute marginal likelihood (approximation for classification case)
      %
      marginal	= dataLikely - 0.5*(logdetH - sum(log(alpha_used)) + ...
                       (w(useful).^2)'*alpha_used);

    % output diagnostic info
%     if (LAST_IT)
      fprintf('iter = %d L = %0.3f Gamma = %0.2f (nz = %d)\n',...
         i, marginal, sum(gamma), sum(useful));
%     end;

      if ~LAST_IT
        % 
        % alpha and beta re-estimation on all but last iteration
        % (only update the posterior statistics the last time around)
        % 
        logAlpha		= log(alpha(useful));
        %
        % Alpha re-estimation
        % 
        % This will be much improved in the subsequent SB2 library
        % 
        % MacKay-style update for alpha given in original NIPS paper
        % 
        alpha(useful)	= gamma ./ w(useful).^2;
        %
        % Terminate if the largest alpha change is smaller than threshold
        % 
        au		= alpha(useful);
        maxDAlpha	= max(abs(logAlpha(au~=0)-log(au(au~=0))));
        if maxDAlpha<MIN_DELTA_LOGALPHA
          LAST_IT	= 1;
          fprintf('Terminating as change in log alpha stagnates...\n');
        end
      else
        % Its the last iteration due to termination, leave outer loop
        break;	% that's all folks!
      end
    end
    %
    % Tidy up return values
    % 
    weights	= w(useful);
    used	= find(useful);

    if ~LAST_IT
      fprintf('Terminating as max number of iterations reached..\n');
    end
    fprintf('Number of non zero parameters is %d\n', length(weights));
end