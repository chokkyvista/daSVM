% MCMC sampling for (binary linear) SVM
% for theoretical and technical details, please refer to the paper
% "Data Augmentation for Support Vector Machines" by Nicholas G. Polson and Steven L. Scotty, published in Bayesian Analysis (2011)
% 
% inputs:
% X - data matrix (K*N, data stored column-wisely)
% y - label vector (N*1, '-1' for negative, '1' for positive)
% lambda - regularization constant
% ell - margin parameter (usually would be 1)
% nepoch - number of epochs in Gibbs sampling
% 
% output: 
% w - the optimal weight vector (w.r.t. the objective function) during the sampling process
% 
% WARNING: lambda should be TWICE the normal case!
%          (check the objective function in fobj.m)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [w] = mcsvm(X, y, lambda, ell, nepoch)
if islogical(y)
    y = 2*y - 1;
end

[k, n] = size(X);
invsigma = diag(lambda*ones(k,1));
mu = invsigma\(X*y);
R = choll(invsigma);

wopt = mu;
fopt = fobj(X,y,wopt,lambda,ell);
for i = 1:nepoch
    w = mu + R\randn(k, 1);
    invgamma = invnrnd(1./abs(ell-y'.*(w'*X))', 1, n);
    tig = sqrt(invgamma)';
    sX = X.*tig(ones(k,1),:);
    invsigma = sX*sX'; % to save memory usage (as compared with: X*diag(invlambda)*X')
    invsigma(1:k+1:k*k) = invsigma(1:k+1:k*k) + lambda; % add lambda to diagonal entries
    R = choll(invsigma);
    mu = R\(R'\(X*(y.*(1+ell*invgamma)))); % to save computation time (as compared with: invsigma\(X*(y.*(1+ell*invgamma))))
    
    ftemp = fobj(X,y,w,lambda,ell);
    if ftemp < fopt
        fopt = ftemp;
        wopt = w;
    end
    fprintf('%d: fobj = %.4f\n', i, ftemp);
end
w = wopt;

end