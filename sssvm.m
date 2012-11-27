% MCMC sampling for (binary linear) SVM under spike-and-slab priors
% for theoretical and technical details, please refer to the paper
% "Data Augmentation for Support Vector Machines" by Nicholas G. Polson and Steven L. Scotty, published in Bayesian Analysis (2011)
% 
% inputs:
% X - data matrix (K*N, data stored column-wisely)
% y - label vector (N*1, '-1' for negative, '1' for positive)
% lambda - regularization constant
% ell - margin parameter (usually would be 1)
% nepoch - number of epochs in Gibbs sampling
% pi - parameters of the spike-and-slab prior (scalar or K*1 vector)
% 
% output: 
% w - the optimal weight vector (w.r.t. the objective function) during the sampling process
% 
% WARNING: lambda should be TWICE the normal case!
%          (check the objective function in fobj.m)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [w] = sssvm(X, y, lambda, ell, nepoch, pi)
if islogical(y)
    y = 2*y - 1;
end

[k, n] = size(X);
if isscalar(pi)
    pi = pi*ones(k,1);
end
invsigma = diag(lambda*ones(k,1));
Xmym1pil = X*y;
mu = invsigma\Xmym1pil;
gamma = rand(k,1)<pi;
R = choll(invsigma(gamma,gamma));
mu_ = R\(R'\Xmym1pil(gamma));

wopt = mu;
fopt = fobj(X,y,wopt,lambda,ell);
w = zeros(k, 1);
for i = 1:nepoch
    % sampling 'w'
    w(~gamma) = 0;
    w(gamma) = mu_ + R\randn(nnz(gamma), 1);
    
    ftemp = fobj(X,y,w,lambda,ell);
    if ftemp < fopt
        fopt = ftemp;
        wopt = w;
    end
    fprintf('%d: fobj = %.4f, |gamma|=%d\n', i, ftemp, nnz(gamma));
    
    % sampling 'gamma'
    for j = 1:k
        gamma(j) = true;
        invsigma_1 = invsigma(gamma,gamma);
        R1 = choll(invsigma_1);
        tismm_1 = Xmym1pil(gamma);
        mu_1 = R1\(R1'\tismm_1);
        
        gamma(j) = false;
        invsigma_0 = invsigma(gamma,gamma);
        R0 = choll(invsigma_0);
        tismm_0 = Xmym1pil(gamma);
        mu_0 = R0\(R0'\tismm_0);

        gamma(j) = rand*(1+pi(j)/(1-pi(j))*prod([diag(R0);1]./diag(R1))*exp(0.5*(mu_1'*tismm_1-mu_0'*tismm_0))) > 1;
%         gamma(j) = rand*(1+pi(j)/(1-pi(j))*sqrt(det(invsigma_0)/det(invsigma_1))*exp(0.5*(mu_1'*tismm_1-mu_0'*tismm_0))) > 1;
    end
    
    % sampling 'invlambda'
    invlambda = invnrnd(1./abs(ell-y'.*(w'*X))', 1, n);
    Xmym1pil = X*(y.*(1+ell*invlambda));
    
    til = sqrt(invlambda)';
    sX = X.*til(ones(k,1),:);
    invsigma = sX*sX'; % to save memory usage (as compared with: X*diag(invlambda)*X')
    invsigma(1:k+1:k*k) = invsigma(1:k+1:k*k) + lambda; % add lambda to diagonal entries
    invsigma_ = invsigma(gamma,gamma);
    R = choll(invsigma_);
    mu_ = R\(R'\Xmym1pil(gamma));
end
w = wopt;

end