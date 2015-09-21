close all;
clear all;

global N START_POINT END_POINT STEP SAMPLES MAX_P;

% Parameters
N = 128;
SIM_ID = 668;
START_POINT = 2;
END_POINT = N;
STEP = 1; % Sampling interval for K
%simMax = 50; %50
SAMPLES = numel(START_POINT:STEP:END_POINT);
MAX_P = 1;
% Intrinsic beliefs
%s = rand(N,1);
% Adjacency Matrix
A = gnpConnected(N, 0.2);
%% Run Simulation
gcp;
opinionsT = [];
parfor z = 1:SAMPLES
    kTable = START_POINT:STEP:END_POINT;
    K = kTable(z);
    fprintf(1,'K: %d\n',K);
    % Run
    [~,op] = kNN_NOMEM(A, s, K, 10e3, 1e-5); % Can Simulate kNN2 too
    opinionsT = [opinionsT; op'];
end
clear op;

%% Create surface plot
%Downsample if needed
opinionsT = downsample(opinionsT',5);
opinionsT = opinionsT';
SAMPLES = ceil(SAMPLES/5);
STEP = 5;
%% Create surface plot
figure;
opinionsT = opinionsT';
xSize = size(opinionsT, 2); % Different Values
HIST_BINS = 40; % Divisions of the [0,1] interval for the histogram
freqData = [];
for i = 1:xSize
    h = histogram(opinionsT(:,i), [0 1], 'Normalization', 'probability');
    h.NumBins = HIST_BINS;
    freqData = [freqData; h.Values];
end
%%
[X, Y] = meshgrid(1:xSize, 1:HIST_BINS);
surf(X,Y,freqData');
axis tight;
ax = gca;
ax.XTick = 1:20:SAMPLES;
ax.XTickLabel = START_POINT:STEP*20:END_POINT;
ax.YTick = [1 10 20 30 40]; % Change this if you change HIST_BINS
ax.YTickLabel = [0 0.25 0.5 0.75 1];

%% Save
save([num2str(SIM_ID) '_data.mat']);