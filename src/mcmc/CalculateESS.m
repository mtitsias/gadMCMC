function [ ESS ] = CalculateESS( Samples, MaxLag )

% Samples is a NumOfSamples x NumOfParameters matrix

[NumOfSamples, NumOfParameters] = size(Samples);

% Calculate empirical autocovariance
ACs = zeros(NumOfSamples, NumOfParameters);
for i = 1:NumOfParameters
    ACs(:,i) = autocorr(Samples(:,i),MaxLag); % Needs statistical toolbox for MATLAB
end



% Calculate Gammas from the autocorrelations
Gamma = zeros(fix(size(ACs,1)/2), NumOfParameters);
for j = 1:((size(ACs,1)/2))
    Gamma(j,:) = ACs(2*j-1,:) + ACs(2*j,:);
end 


% Calculate the initial monotone convergence estimator
% -> Gamma(j,i) is min of preceding values
for j = 2:((size(ACs,1)/2))
    Gamma(j,:) = min(Gamma(j-1:j,:),[],1);
end


MonoEst = zeros(1,NumOfParameters);
for i = 1:NumOfParameters
    % Get indices of all Gammas greater than 0
    PosGammas = find(Gamma(:,i)>0);
    % Sum over all positive Gammas
    MonoEst(i) = -ACs(1,i) + 2*sum(Gamma(1:length(PosGammas),i));
    
    % MonoEst cannot be less than 1 - fix for when lag 2 corrs < 0
    if MonoEst(i) < 1
        MonoEst(i) = 1;
    end
end

ESS = NumOfSamples./MonoEst;

end
