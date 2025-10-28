%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Requires: https://github.com/anders-s-olsen/psilocybin_dynamic_FC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

ROI_sig = readmatrix('time_series.txt');
ROI_sig = ROI_sig(:,1:5);

allpair = 0; parallel = 0;
tic
[ H,R,Theta,X ] = DCC_X( ROI_sig, allpair, parallel );
toc

dcc = R;
save('/home/mibur/comet/tests/MATLAB/dcc.mat','dcc');