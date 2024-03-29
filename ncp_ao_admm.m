function [P,Out] = ncp_ao_admm(X,R,varargin)
%NCP_AO_ADMM: Nonnegative CANDECOMP/PARAFAC tensor decomposition by
% alternating direction method of multipliers in block coordinate descent
% framework.
%
% min 0.5*||M - A_1\circ...\circ A_N||_F^2 + r(A)
% subject to A_1>=0, ..., A_N>=0
%
% input: 
%       X: input nonnegative tensor
%       R: estimated rank (each A_i has r columns); require exact or moderate overestimates
%       varargin.
%           'tol' - tolerance for relative change of function value, default: 1e-4
%           'maxiters' - max number of iterations, default: 500
%           'maxtime' - max running time, default: 1000
%           'dimorder' - Order to loop through dimensions {1:ndims(A)}
%           'init' - Initial guess [{'random'}|'nvecs'|cell array]
%           'printitn' - Print fit every n iterations; 0 for no printing {1}
%           'stop' - Stopping condition. 0 for stopping by maxtime or
%           maxiters; 1 for change of objective function; 2 for change of
%           data fitting. Default: 2
%           'regularization' - parameters of Frobenius regularization and
%           sparse regularization.
% output:
%       P: nonnegative ktensor
%       Out.
%           iter: number of iterations
%           time: running time at each iteration
%           obj: history of objective values
%           relerr: history of relative objective changes (row 1) and relative residuals (row 2)
%
% Require MATLAB Tensor Toolbox from
% http://www.sandia.gov/~tgkolda/TensorToolbox/
%
% References:
% [1]. Boyd S, Parikh N, Chu E, et al. Distributed optimization and
% statistical learning via the alternating direction method of multipliers.
% Foundations and Trends� in Machine learning, 2011, 3(1): 1-122.
% [2]. Xu Y, Yin W, Wen Z, et al. An alternating direction algorithm for
% matrix completion with nonnegative factors[J]. Frontiers of Mathematics
% in China, 2012, 7(2): 365-384.
% [3]. Huang K, Sidiropoulos N D, Liavas A P. A flexible and efficient
% algorithmic framework for constrained matrix and tensor factorization.
% IEEE Transactions on Signal Processing, 2016, 64(19): 5052-5065.
% 
%
% Author: Deqing Wang
% Email: deqing.wang@foxmail.com
% Website: http://deqing.me/
% Affiliation: Dalian University of Technology, China
%              University of Jyv�skyl�, Finland
% Date: July 20, 2019
%

%% Extract number of dimensions and norm of X.
N = ndims(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxiters',500,@(x) isscalar(x) & x > 0);
params.addParameter('maxtime', 1000,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParameter('regparams',zeros(N,2),@(x) (ismatrix(x) || sum(any(x<0))==0));
params.addParameter('stop', 1, @(x) (isscalar(x) & ismember(x,[0,1,2])));
params.addParameter('printitn',1,@isscalar);
params.parse(varargin{:});

%% Copy from params object
tol = params.Results.tol;
maxiters = params.Results.maxiters;
maxtime = params.Results.maxtime;
dimorder = params.Results.dimorder;
init = params.Results.init;
regparams = params.Results.regparams;
stop = params.Results.stop;
printitn = params.Results.printitn;

%% Error checking 
% Error checking on maxiters
if maxiters < 0
    error('OPTS.maxiters must be positive');
end

% Error checking on dimorder
if ~isequal(1:N,sort(dimorder))
    error('OPTS.dimorder must include all elements from 1 to ndims(X)');
end

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(1:end)
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            Uinit{n} = max(0,randn(size(X,n),R)); % randomly generate each factor
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(1:end)
            k = min(R,size(X,n)-2);
            fprintf('  Computing %d leading e-vectors for factor %d.\n',k,n);
            Uinit{n} = abs(nvecs(X,n,k));
            if (k < R)
              Uinit{n} = [Uinit{n} rand(size(X,n),R-k)]; 
            end
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit0 = 0;

% Initial object value
obj0=0.5*normX^2;

% Initialize the dual factors
U_dual=cell(N,1);
for n = dimorder(1:end)
    U_dual{n}=zeros(size(U{n}));
end

% Normalize factors
Xnormpower=normX^(1/N);
for n = dimorder(1:end)
    U{n}=U{n}/norm(U{n},'fro')*Xnormpower;
end

% Tolerance
tolU = 0.01*ones(N,1);

nstall = 0;

%% Main Loop: Iterate until convergence
start_time = tic;
if printitn>=0
    fprintf('\nNonnegative CANDECOMP/PARAFAC using AO-ADMM:\n');
end
if printitn==0, fprintf('Iteration:      '); end
for iter = 1:maxiters
    if printitn==0, fprintf('\b\b\b\b\b\b%5i\n',iter); end
    
    iterU0=zeros(N,1);
    % Iterate over all N modes of the tensor
    for n = dimorder(1:end)

        % Compute the matrix of coefficients for linear system
        BtB = ones(R,R);
        for i = [1:n-1,n+1:N]
            BtB = BtB .* (U{i}'*U{i});
        end
        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        MTTKRP = mttkrp(X,U,n);

        % Solve non-negative least squares problem
        [Unew,U_dual_new,iterU]=solver_admm_scaled_Huang2016(MTTKRP',BtB,U{n}',U_dual{n}','tol',tolU(n),'beta',regparams(n,2));
        if iterU==1
            tolU(n) = 0.1*tolU(n);
        end
        
        iterU0(n)=iterU;
        U{n} = Unew';
        U_dual{n} = U_dual_new';
    end
    
    % --- diagnostics, reporting, stopping checks ---
    % Initial objective function value
    obj = 0.5*( normX^2 - 2 * sum(sum(U{n}.*MTTKRP)) +...
        sum(sum((U{n}'*U{n}).*BtB)));    
    % After above step, normresidual equals to 
    % 0.5*( normX^2 - 2 * innerprod(X,P) + norm(P)^2 ), where P = ktensor(U).
    
    % Norm of residual value.
    normresidual = sqrt(2*obj);
    % After above step, normresidual equals to
    % sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) ), where P = ktensor(U).
    
    % Objective function value
    for n = dimorder(1:end)
        if regparams(n,2)>0 % L1-norm sparse regularization
            obj = obj + regparams(n,2) * sum(abs(U{n}(:)));
        end
    end
    
    % Compute performance evaluation values
    relerr1 = abs(obj-obj0)/(obj0+1); % relative objective change
    relerr2 = (normresidual / normX); % relative residual
    fit = 1 - relerr2; %fraction explained by model
    fitchange = abs(fit0 - fit);
    current_time=toc(start_time);
    
    % Record performance evaluation values
    Out.obj(iter) = obj;
    Out.relerr(1,iter) = relerr1;
    Out.relerr(2,iter) = relerr2;
    Out.time(iter) = current_time;
    
    % Display performance evaluation values
    if printitn>0
        if mod(iter,printitn)==0
            printout1 = sprintf(' Iter %2d: fit = %e fitdelta = %7.1e, inner iter: ',...
                iter, fit, fitchange);
            printout2 = mat2str(iterU0);
            fprintf([printout1 printout2(2:end-1) '\n']);
        end
    end

    % Check stopping criterion
    if stop == 1
        crit = (relerr1<tol);
    elseif stop == 2
        crit = (fitchange < tol);
    else
        crit = 0;
    end
    if crit; nstall = nstall+1; else, nstall = 0; end
    if (nstall >= 3 || relerr2 < tol) && stop == 1; break; end
    if iter > 1 && nstall >= 3 && stop == 2; break; end
    if current_time > maxtime; break; end
    
    % Update previous object function value
    obj0 = obj;
    fit0 = fit;
end

%% Clean up final result
P = ktensor(U);

if printitn>0
  fprintf(' Final fit = %e \n', fit);
end

Out.iter=iter;

return;

