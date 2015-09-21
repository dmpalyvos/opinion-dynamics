%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test if clustering appears when we change the å parameter of HK Local Eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%close all;
%clear all;
global SIM_ID SAVE_GRAPH;

% Parameters
SAVE_GRAPH = false;

N = 128;
SIM_ID = 664;
END_POINT = N;
simMax = 96; %50
START_POINT = 2;
STEP = 5; % Sampling interval for K
SAMPLES = numel(START_POINT:STEP:END_POINT);
opinionsT = [];
% Intrinsic beliefs
s = rand(N,1);

%% Run Simulation
gcp;
for z = 1:SAMPLES
    opSim = [];
    kTable = START_POINT:STEP:END_POINT;
    K = kTable(z);
    fprintf(1,'K: %d\n',K);
    parfor i = 1:simMax
        pTable = linspace(0,1,simMax);
        A = gnpConnected(N,pTable(i));
        [~, op] = hkLocalKNN_NOMEM(A, s, K, 10e3, 1e-5); % Can simulate kNN2 too
        opSim = [opSim; op];
    end
    opinionsT = [opinionsT opSim];
end

%% Create surface plot
%%Downsample if needed
%opinionsT = downsample(opinionsT',5);
%opinionsT = opinionsT';
%SAMPLES = ceil(SAMPLES/5);
%STEP = 5;
%% Preprocess
handle = figure;
xSize = size(opinionsT, 2); % Different Values
HIST_BINS = 40; % Divisions of the [0,1] interval for the histogram
freqData = [];
for i = 1:xSize
    h = histogram(opinionsT(:,i), [0 1], 'Normalization', 'probability');
    h.NumBins = HIST_BINS;
    freqData = [freqData; h.Values];
end

%% Plot
[X, Y] = meshgrid(1:xSize, 1:HIST_BINS);
surf(X,Y,freqData');
axis tight;
ax = gca;
JUMP_S = 5;
ax.XTick = 1:JUMP_S:SAMPLES;
ax.XTickLabel = START_POINT:STEP*JUMP_S:END_POINT;
ax.YTick = [1 10 20 30 40]; % Change this if you change HIST_BINS
ax.YTickLabel = [0 0.25 0.5 0.75 1];


%% Save
save([num2str(SIM_ID) '_data.mat']);