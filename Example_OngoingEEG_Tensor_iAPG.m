% Ongoing EEG tensor decomposition using iAPG algorithm.
% Author: Deqing Wang
% Email: deqing.wang@foxmail.com
% Website: http://deqing.me/
% Affiliation: Dalian University of Technology, China
%              University of Jyväskylä, Finland
% Date: July 13, 2021
%
% Require MATLAB Tensor Toolbox from
% http://www.sandia.gov/~tgkolda/TensorToolbox/
%
% Reference:
% Deqing Wang, Zheng Chang and Fengyu Cong. Sparse Nonnegative Tensor
% Decomposition Using Proximal Algorithm and Inexact Block Coordinate
% Descent Scheme, Neural Computing and Applications. 2021.
%

%%
clear;
close all

%% Load Tensor Data
load(['OngoingEEG_Data' filesep 'OngoingEEG_Tensor']);
load(['OngoingEEG_Data' filesep 'mirLongTermFeatures']);
load(['OngoingEEG_Data' filesep 'chanlocs64']);

%% Ongoing EEG tensor description
% ChannelMode             = 64;	% Representing 64 channels
% FrequencyMode           = 146;	% Representing 1-30Hz
% TimeMode                = 510;	% Representing 8.5 minutes

%% Preparation of tensor decomposition
TensorTrue = tensor(OngoingEEG_Tensor); % The third-order tensor
N = ndims(TensorTrue);

% Tensor Decomposition Parameters
R = 40; % The pre-defined number of components

% Tensor Decomposition Parameters
% Parameter format: [0, beta_n]
% beta_n: sparse regularization parameter
%
% Example values for the ongoing EEG: SparseParam = 0, 1e5, 5e5, 10e5, 15e5, 20e5
SparseParam = 0;
RegParams = repmat([0.000 SparseParam],N,1);

%% Start of the NCP tensor decomposition
rng('shuffle');
[A,Out] = ncp_iapg_dynamic_tolerance(TensorTrue,R,'maxiters',99999,'tol',1e-8,...
    'init','random','regparams',RegParams,'printitn',1,...
    'maxtime',120,'stop',2);

%%
fprintf('\n');
fprintf('Total iteration is %d.\n',Out.iter);
fprintf('Elapsed time is %4.1f seconds.\n',Out.time(end));
fprintf('Objective function value is %.4e\n',Out.obj(end));
fprintf('Solution relative error = %4.4f\n\n',Out.relerr(2,end));

%%
% Nonzero column number
FactorNonzeroNum=zeros(ndims(tensor(TensorTrue)),1);
for i=1:ndims(tensor(TensorTrue))
    Factor_SparseIndex=(A.U{i}>1e-6);
    Factor_Sparse=A.U{i}.*double(Factor_SparseIndex);
    SumOfFactor=sum(Factor_Sparse,1);
    FactorNonzeroNum(i)=sum(SumOfFactor~=0);
end
% Reporting
fprintf('Nonzero Components Number:\n');
for i=1:ndims(tensor(TensorTrue))
    fprintf('Nonzero Components Number of A{%d}:\t%d\n',i,FactorNonzeroNum(i,1));
end
fprintf('\n');

%% Plot objective function value
figure;
semilogy(Out.time,Out.obj,'LineWidth',2);
grid on
title('Objective Function Value');
xlabel('Time(s)');
ylabel('Objective Function Value');

