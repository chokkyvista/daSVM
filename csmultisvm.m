% Solve Crammer & Singer multi-class SVMs via data augmentation.
% For theoretical and technical details, please refer to our paper
% "Fast Parallel SVM using Data Augmentation" (to appear)
% 
% inputs:
% X - data matrix (N*K, data stored row-wisely)
% y - label vector (N*1, ranging from 1 to M)
% XX, yy - same for test or validation data
% nepoch - number of epochs in Gibbs sampling
% nsub - number of subiterations in each epoch
% C - regularization constant (check 'fobj')
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
% 1. X and XX are different from those in MCSVM! (transposed)
% 2. The EM implementation is currently not numerically stable!
% 
% See also MCSVM
% 
% Written by Minjie Xu (chokkyvista06@gmail.com) based on an initial
% implementation by Hugh Perkins (hughperkins@gmail.com)

function [w, fvals, mfval, accu, macc, iw] = csmultisvm(X, y, XX, yy, nepoch, nsub, C, burnin, emormc, iw)
% convert y values to 2 if equal -1
if min(y) == -1
    y(y==-1) = 2;
end
if min(yy) == -1
    yy(yy==-1) = 2;
end

[N,K] = size(X);
M = max(y);
fprintf('N = %i, K = %i, M = %i\n', N, K, M);
algs = {'EM', 'MC'};
fprintf('%s, C = %.4f, iter = %d(%d), burnin = %d\n', ...
    algs{emormc+1}, C, nepoch, nsub, burnin*emormc);

lambda = 2/C;
svthreshold = 0.0001;

if ~exist('iw', 'var')
    iw = randn(K, M);
end
w = iw;
mw = zeros(K,M);
yind = sub2ind([N,M],(1:N)',y);
delta = ones(N,M);  % N x M
delta(yind) = 0;
Xw = X * w;

stats = arrayfun(@(x)zeros(nepoch,1), zeros(1,4), 'UniformOutput', false);
[fvals, mfval, accu, macc] = stats{:};

minfval = inf;
cnvg = 0;
for iter = 1:nepoch
    fprintf('%2i: ', iter);
    
    for m = 1:M
        Xwm = Xw(:, m);
        zetam = Xw + delta;
        zetam = zetam(:, [1:(m-1),(m+1):M]); % remove m column
        zetam = max(zetam, [], 2);
        rhom = zetam - delta(:,m);
        for subit = 1:nsub
            gammam = abs(rhom - Xwm);
            gammam = max(svthreshold, gammam);
            if emormc
                invgammam = invnrnd(1./gammam, 1);
            else
                invgammam = 1./gammam;
            end
            
            tmp = sqrt(invgammam);
            tmp = tmp(:, ones(1,K)).*X;
            invSigma = tmp'*tmp;
            invSigma(1:K+1:K*K) = invSigma(1:K+1:K*K) + lambda;
            
            b = rhom.*invgammam + (2*(y==m)-1);
            b = X' * b;
            R = chol(invSigma);
            mu = R\(R'\b);
            if emormc
                w(:,m) = mu + R\randn(K,1);
            else
                w(:,m) = mu;
            end
            Xwm = X * w(:,m);
            Xw(:, m) = Xwm;
        end
    end
    
    [~, ty] = max(XX*w, [], 2);
    accu(iter) = mean(ty==yy);
    fvals(iter) = fobj(lambda,w,Xw,delta,yind,M,N);
    
    if emormc && iter > burnin
        mw = mw + (w-mw)./(iter-burnin);
        [~, ty] = max(XX*mw, [], 2);
        macc(iter) = mean(ty==yy);
        mfval(iter) = fobj(lambda,mw,Xw,delta,yind,M,N);
    else
        macc(iter) = nan;
        mfval(iter) = nan;
    end
    
    fprintf('acc = %.4f, macc = %.4f, obj = %.4f, mobj = %.4f\n', ...
        accu(iter), macc(iter), fvals(iter), mfval(iter));
    if abs(minfval-fvals(iter)) <= 0.0001
        cnvg = cnvg + 1;
    else
        cnvg = 0;
    end
    if cnvg >= 10
        break;
    end
    minfval = min(fvals(iter), minfval);
end
if emormc
    w = mw;
end

end

function objective = fobj(lambda,w,Xw,delta,yind,M,N)
rglzr = 0.5 * sum(w(:).^2);
Xwy = Xw(yind);
mgncst = Xw - repmat(Xwy,1,M) + delta;
loss = sum(max(mgncst, [], 2));
objective = (lambda*rglzr + 2*loss)/N;

end
