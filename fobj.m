% objective function in the binary linear SVM problem
% 
% inputs:
% X - data matrix (K*N, data stored column-wisely)
% y - label vector (N*1, '-1' for negative, '1' for positive)
% w - weight vector (K*1)
% lambda - regularization constant
% ell - margin parameter (usually would be 1)
% 
% WARNING: lambda should be TWICE the normal case!
%          (check line 16)
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function f = fobj(X, y, w, lambda, ell)
f = 0.5*lambda*sum(w.^2) + 2*sum(max(0, ell-y'.*(w'*X)));

end