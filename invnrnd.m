% sample from Inverse Gaussian distribution
% http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Generating_random_variates_from_an_inverse-Gaussian_distribution
% 
% inputs:
% mu, lambda - parameters of the IG distribution
%   supported format:
%     1. two arrays of exactly the same size
%     2. mu is array & lambda is scalar (lambda is reused for each sample)
% n - number of samples per parameter setting (defaults to 1)
% 
% output: 
% R - returned samples ( size(R)==size(squeeze(zeros([size(mu),n]))) )
% 
% WARNING: returns NAN where mu==inf
% 
% Written by Minjie Xu (chokkyvista06@gmail.com)

function R = invnrnd(mu, lambda, n)
if ~exist('n', 'var')
    n = 1;
end
if ~isscalar(lambda)
    assert(all(size(lambda)==size(mu)));
    lambda = squeeze(repmat(lambda, [ones(1,ndims(mu)),n]));
end
mu = squeeze(repmat(mu, [ones(1,ndims(mu)),n]));

nu = randn(size(mu));
y = nu.^2;
muy = y.*mu;
x = mu.*(1+(muy-sqrt(muy.*(4*lambda+muy)))./(2*lambda));
R = x;
elsind = rand(size(mu)).*(mu+x) > mu;
R(elsind) = (mu(elsind).^2)./x(elsind);

end