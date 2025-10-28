%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Requires: https://github.com/anders-s-olsen/psilocybin_dynamic_FC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

ROI_sig = readmatrix('time_series.txt');
ROI_sig = ROI_sig(:,1:5);

T = {size(ROI_sig,1)}; % one subject with one session of length T
options.flip_eigenvectors = true;

leida = pdfc_compute_eigenvectors(ROI_sig, T, options);

save('/home/mibur/comet/tests/MATLAB/leida.mat','leida');
