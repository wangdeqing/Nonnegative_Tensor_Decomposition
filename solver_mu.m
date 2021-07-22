function [ H, iter ] = solver_mu(WtV, WtW, H, varargin)
%SOLVER_MU Multiplicative update
%   minimize (1/2)*|| V - W*H ||^2 + r(H),
%   s.t. H >= 0.
%
% Author: Deqing Wang
% Email: deqing.wang@foxmail.com
% Website: http://deqing.me/
% Affiliation: Dalian University of Technology, China
%              University of Jyväskylä, Finland
% Date: July 19, 2019
%
%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('maxiters',5, @(x) isscalar(x) & x > 0);
params.addParameter('beta',0, @(x) isscalar(x) & x >= 0);
params.addParameter('tol',1e-2,@isscalar);
params.parse(varargin{:});

%% Copy from params object
maxiters = params.Results.maxiters;
beta = params.Results.beta;
tol = params.Results.tol;

%%
% Prevent the numerator to be zero
WtV = WtV + eps;
%%
for iter = 1:maxiters
    
    H0 = H;
    % Multiplicative Update
    WtWH = WtW*H;
    % Prevent the denominator to be zero
    if beta > 0
        WtWH = WtWH + beta;
    else
        WtWH = WtWH + eps;
    end
    
    H = H .* WtV;
    H = H ./ WtWH;

    Rh = H0 - H;
    
    if norm(Rh(:)) < tol*norm(H(:))
        break
    end
    
end
end
