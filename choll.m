% error-free Cholesky factorization
% same as chol(A) if A is positive definite, 
% otherwise, try adding to A the positive definite diagnoal matrix mI multiple times
% until Cholesky factorization is applicable
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function [q,P] = choll(A)
P = A;
m = 1e-12;
[q,s] = chol(P);
diagind = 1:size(P,1)+1:numel(P);
cc = 0;
while s~=0
    P(diagind) = P(diagind) + m;
    cc = cc + 1;
    [q,s] = chol(P);
end
if cc > 0
    warning('Augmented %d times by %.4f\n', cc, cc*m);
end

end