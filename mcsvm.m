% MCMC sampling for (binary linear) SVMs.
% For theoretical and technical details, please refer to the paper
% "Data Augmentation for Support Vector Machines" by Nicholas G. Polson and Steven L. Scotty, published in Bayesian Analysis (2011)
% 
% inputs:
% X - data matrix (K*N, data stored column-wisely)
% y - label vector (N*1, '-1' for negative, '1' for positive)
% XX, yy - same for test or validation data
% lambda - regularization constant
% ell - margin parameter (usually would be 1)
% nepoch - number of epochs in Gibbs sampling
% burnin - discard the first 'burnin' samples
% emormc - use EM (0) or MCMC sampling (1)
% iw - initial value of w
% 
% output: 
% w - the last sample in EM or the averaged sample in MCMC
% fvals - objective function values of single samples during the iteration
% accu - test or validation accuracy of single samples during the iteration
% mfval, macc - same for averaged samples (after burnin)
% iw - initial value of w
% 
% WARNING: 
% 1. lambda should be TWICE the normal case!
%    (check the objective function in fobj.m)
% 2. The EM implementation is currently NOT numerically stable!
%    (check Sec. 3.2 in the paper)
% 
% See also FOBJ, INVNRND
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [w, fvals, mfval, accu, macc, iw] = mcsvm(X, y, XX, yy, lambda, ell, nepoch, burnin, emormc, iw)
if islogical(y)
    y = 2*y - 1;
end
yy = (yy==1);

[k, n] = size(X);
fprintf('N = %i, K = %i\n', n, k);
algs = {'EM', 'MC'};
fprintf('%s, lambda = %.4f, i = %d, burnin = %d\n', ...
    algs{emormc+1}, lambda, nepoch, burnin*emormc);

if ~exist('iw', 'var')
    invsigma = diag(lambda*ones(k,1));
    mu = invsigma\(X*y);
    R = choll(invsigma);
    iw = mu + R\randn(k, 1);
end
w = iw;
mw = zeros(k, 1);

stats = arrayfun(@(x)zeros(nepoch,1), zeros(1,4), 'UniformOutput', false);
[fvals, mfval, accu, macc] = stats{:};

minfval = inf;
cnvg = 0;
for i = 1:nepoch
    fprintf('%2i: ', i);
    
    invgamma = 1./abs(ell-y'.*(w'*X))';
    indinf = isinf(invgamma);
    if any(indinf(:))
        invgamma(indinf) = max(invgamma(~indinf)).^2; % avoid infinite invgamma
    end
    if emormc
        invgamma = invnrnd(invgamma, 1, n);
    end
    tig = sqrt(invgamma)';
    sX = X.*tig(ones(k,1),:);
    invsigma = sX*sX'; % to save memory usage (as compared with: X*diag(invlambda)*X')
    invsigma(1:k+1:k*k) = invsigma(1:k+1:k*k) + lambda; % add lambda to diagonal entries
    R = choll(invsigma);
    mu = R\(R'\(X*(y.*(1+ell*invgamma)))); % to save computation time (as compared with: invsigma\(X*(y.*(1+ell*invgamma))))
    
    if emormc
        w = mu + R\randn(k, 1);
    else
        w = mu;
    end
    
    ty = XX'*w >= 0;
    accu(i) = mean(ty==yy);
    fvals(i) = fobj(X,y,w,lambda,ell);
    
    if emormc && i > burnin
        mw = mw + (w-mw)./(i-burnin);
        ty = XX'*mw >= 0;
        macc(i) = mean(ty==yy);
        mfval(i) = fobj(X,y,mw,lambda,ell);
    else
        macc(i) = nan;
        mfval(i) = nan;
    end
    
    fprintf('acc = %.4f, macc = %.4f, obj = %.4f, mobj = %.4f\n', ...
        accu(i), macc(i), fvals(i), mfval(i));
    if abs(minfval-fvals(i)) <= 0.0001*n
        cnvg = cnvg + 1;
    else
        cnvg = 0;
    end
    if cnvg >= 10
        break;
    end
    minfval = min(fvals(i), minfval);
end
if emormc
    w = mw;
end

end