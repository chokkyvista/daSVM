% sample from Inverse Gaussian distribution
% http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Generating_random_variates_from_an_inverse-Gaussian_distribution
% 
% inputs:
% mu, lambda - parameters of the IG distribution
%   supported format:
%     1. 2 scalars (use 'n' to indicate number of samples or omit it to do sampling only once)
%     2. 2 column vectors of same length ('R' will be of same length as 'mu' & 'lambda')
%     3. mu is column vector & lambda is scalar ('R' will be of same length as 'mu', 'lambda' is reused for each sample)
% n - when specified, number of elements in the output R
% 
% output: 
% R - column vector of returned samples
% 
% WARNING: returns NAN when mu==inf
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function R = invnrnd(mu, lambda, n)
if ~exist('n', 'var')
    n = numel(mu);
end
if numel(mu) == 1 && n > 1
    mu = mu*ones(n, 1);
end
nu = randn(n, 1);
y = nu.^2;
muy = y.*mu;
x = mu.*(1+(muy-sqrt(muy.*(4*lambda+muy)))./(2*lambda));
R = x;
elsind = rand(n, 1).*(mu+x) > mu;
R(elsind) = (mu(elsind).^2)./x(elsind);

end