%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% As implemented in: https://github.com/brain-networks/edge-ts/blob/master/main.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

ROI_sig = readmatrix('time_series.txt');
ROI_sig = ROI_sig(:,1:5);

% z-score time series
ts = zscore(ROI_sig);

% get dimensions
[ntime,nnodes] = size(ts);

% calculate number of edges
nedges = nnodes*(nnodes - 1)/2;

% indices of unique edges (upper triangle)
[u,v] = find(triu(ones(nnodes),1).');  % NOTE: Added transpose here to match Python

idx = (v - 1)*nnodes + u;

%% calculate static fc
fc = corr(ts);

%% generate edge time series
ets = ts(:,u).*ts(:,v);

save('/home/mibur/comet/tests/MATLAB/ets.mat','ets');