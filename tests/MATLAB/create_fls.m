%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Requires: https://github.com/guorongwu/DynamicBC/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% --- Parameters ---
ts_file   = 'time_series.txt';   % input time series
mu        = 100;                 % regularisation parameter
tmp_file  = '/home/mibur/comet/tests/MATLAB/FCM/DFC.mat'; % intermediate output from function
out_file  = '/home/mibur/comet/tests/MATLAB/fls.mat'; % final clean array

%% --- Run dynamic FC (saves intermediate struct in ./FCM/DFC.mat) ---
ROI_sig = readmatrix(ts_file);
ROI_sig = ROI_sig(:,1:5);
save_info.flag_nii     = false;
save_info.save_dir     = pwd;
save_info.nii_mat_name = fullfile(pwd,'DFC.mat');

fprintf('Running DynamicBC_fls_FC with mu=%d on %s...\n', mu, ts_file);
DynamicBC_fls_FC(ROI_sig, mu, save_info);
fprintf('Intermediate results saved to %s\n', tmp_file);

%% --- Convert FCM struct -> 3D array ---
S = load(tmp_file);   % loads struct FCM
FCM = S.FCM;



P   = size(FCM.Matrix{1},1);   % number of ROIs
T   = numel(FCM.Matrix);       % number of time points

P, T

fls = zeros(P,P,T);
for t = 1:T
    fls(:,:,t) = FCM.Matrix{t};
end

%% --- Save clean array ---
save(out_file,'fls');
