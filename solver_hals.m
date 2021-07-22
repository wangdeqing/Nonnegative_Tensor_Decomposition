function [ H, iter ] = solver_hals(WtV, WtW, H, outer_iter, varargin)
%SOLVER_HALS hierarchical alternating least squares
%   minimize (1/2)*|| V - W*H ||^2 + r(H),
%   s.t. H >= 0.
%
% Author: Deqing Wang
% Email: deqing.wang@foxmail.com
% Website: http://deqing.me/
% Affiliation: Dalian University of Technology, China
%              University of Jyväskylä, Finland
% Date: July 15, 2019
%
%%
[R,~] = size(H);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('maxiters',5, @(x) isscalar(x) & x > 0);
params.addParameter('alpha',0, @(x) isscalar(x) & x >= 0);
params.addParameter('tol',1e-2,@isscalar);
params.parse(varargin{:});

%% Copy from params object
maxiters = params.Results.maxiters;
alpha = params.Results.alpha;
tol = params.Results.tol;

%%

for iter = 1:maxiters
    H0 = H;
    
    % Update H using HALS method.
    for r=1:R
        H(r,:)=H(r,:) + (WtV(r,:) - WtW(r,:)*H)/(WtW(r,r) + alpha);
        
        if outer_iter<=5
            % Empirically, using eps in the first few iternations can make
            % the algorithm more stable.
            H(r,:)=max(H(r,:),eps);
        else
            H(r,:)=max(H(r,:),0);
        end
    end
    
    r = H - H0;
    if norm(r(:)) < tol*norm(H(:))
        break
    end

end
end
