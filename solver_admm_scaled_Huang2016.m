function [ H_hat, Psi, iter ] = solver_admm_scaled_Huang2016(WtV, WtW, H_hat, Psi, varargin)
% Scaled ADMM to solve
%   minimize (1/2)*|| V - W*H ||^2 + r(H),
%   s.t. H >= 0.
%
% Author: Deqing Wang
% Email: deqing.wang@foxmail.com
% Website: http://deqing.me/
% Affiliation: Dalian University of Technology, China
%              University of Jyväskylä, Finland
% Date: April 17, 2019
%
%%
[R,~] = size(H_hat);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('tol',1e-2,@isscalar);
params.addParameter('maxiters',5,@(x) isscalar(x) & x > 0);
params.addParameter('beta',0,@isscalar);
params.parse(varargin{:});

%% Copy from params object
tol = params.Results.tol;
maxiters = params.Results.maxiters;
beta = params.Results.beta;

%%
rho = trace(WtW)/R;
rho = max(rho, 1e-12);% Guarantee Denominator to be positive definite.
Denominator = WtW + rho*eye(R);
beta_to_rho = beta/rho;

L = chol(Denominator, 'lower');

for iter = 1:maxiters
    H_hat0 = H_hat;
    
    H = L'\ ( L\ ( WtV + rho*(H_hat-Psi) ) );
    H_hat = max(0, H + Psi - beta_to_rho);% Proximal operator
    r = H - H_hat;
    Psi  = Psi + r;
    
    s = H_hat0 - H_hat;
    if norm(r(:)) < tol*norm(H_hat(:)) && norm(s(:)) < tol*norm(Psi(:))
        break
    end
end
end
