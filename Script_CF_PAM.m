% Supplementary Material - S. Bae et al., IEEE Transaction on Biomedical Engineering, 2022
%
% This script is provided to help the readers of IEEE TBME to implement
% the coherence-factor-based passive acoustic mapping (CF-PAM).
%
clear; close all;

gd = gpuDevice();
wait(gd);

% lad RF data
load('stRcvData');
mRcvData = stRcvData.mRcvData;

% transducer info
stTr = stRcvData.stTrans;
% rf data info
stRf = stRcvData.stRf;

% BF image Grid info
% - lateral pixel 
stG.dx = (stTr.aElePos(2)-stTr.aElePos(1));
stG.nXdim = 128;  
stG.aX = (-(stG.nXdim-1)/2:1:(stG.nXdim-1)/2)*stG.dx; % lateral axis [m]
% - axial pixels
nUnitDist = stRf.nSoundSpeed/(2*stRf.nFs);
stG.dz = nUnitDist*16;
stG.aZ = 5e-3:stG.dz:55e-3; % axial axis [m]
stG.nZdim = numel(stG.aZ);

% - number of integration time samples
stG.nTdim = 100000;

rf_nOffsetDelay_m = 0;
rf_nMeter2Pixel = stRf.nFs/stRf.nSoundSpeed; % [pixel/m] = sampling frq/ sound speed

% allocate gpu array for cavitation map of cuda
vCavMap_tzx_gpu    = single(gpuArray(zeros([stG.nTdim,stG.nZdim,stG.nXdim])));

% compile cuda code
bCompile = 0;
if bCompile
    disp('compiling cuda code..');
    system(['nvcc -ptx PAM_CF_v1.cu' ' -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC"' ]);
else
    disp('using existing compiled code...');
end

% setup cuda kernel
disp('setting kernel..');
kPAM = parallel.gpu.CUDAKernel('PAM_CF_v1.ptx','PAM_CF_v1.cu','_PAM_CF');
kPAM.ThreadBlockSize =[1024,1,1];
kPAM.GridSize = [ceil(stG.nTdim/kPAM.ThreadBlockSize(1)),stG.nXdim,stG.nZdim];
setConstantMemory(kPAM,...
    'trans_aElePos',        single(stTr.aElePos),...   % transducer element position in x
    'trans_nNumEle',        int32(stTr.nNumEle),...    % num of txdcr elements
    'rf_nSample',           int32(stRf.nSample),...       % num of RF samples
    'rf_nChannel',          int32(stRf.nChannel),...      % num of channels of RF data
    'rf_nOffsetDelay_m',    single(rf_nOffsetDelay_m),... % [m]  offset of RF data
    'rf_nMeter2Pixel',      single(rf_nMeter2Pixel),...  % [pixel/m]  = sampling frequency / SoundSpeed
    'g_nXdim',              int32(stG.nXdim),...
    'g_nZdim',              int32(stG.nZdim),...
    'g_dx',                 single(stG.dx),...
    'g_dz',                 single(stG.dz),...
    'g_nXstart',            single(stG.aX(1)),...
    'g_nZstart',            single(stG.aZ(1)), ...    
    'g_nTdim',              int32(stG.nTdim));


% transfer RF data to GPU mem
mRfData_sc_gpu = single(gpuArray(mRcvData)); 
wait(gd);    

tic;
% call the kernel
[vCavMap_tzx_gpu] = feval(kPAM, vCavMap_tzx_gpu, mRfData_sc_gpu);
wait(gd);  
% sum over time
mCavMap_zx = squeeze(gather(sum(vCavMap_tzx_gpu,1)));
toc;


% Plot
figure;
mCavMap_norm = (mCavMap_zx-min(mCavMap_zx(:)))/(max(mCavMap_zx(:))-min(mCavMap_zx(:))); % normalization
imagesc(stG.aX*1e3,stG.aZ*1e3,mCavMap_norm); axis equal; axis tight; 
xlabel('x (mm)'); ylabel('z (mm)');
