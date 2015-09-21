%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test if clustering appears when we change the relative size of B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
global SIM_ID NORM_TYPE SAVE_GRAPH;

% Parameters
SIM_ID = 13; % Simulation ID, used for filename id
NORM_TYPE = 2;
SAVE_GRAPH = false;

N = 64;
simMax = 20;

A = gnpConnected(N,0.1);
% Intrinsic beliefs

opinionsT = [];
for bb = logspace(2,-2,40)
    opSim = [];
    parfor i = 1:simMax
        s = rand(N,1);
        B = rand(N,1);
        % Stubborness
        sB = bb*B;
        % Run
        [~, op] = generalizedAsymmetric(A, sB, s, 10e3, 1e-5);
        opSim = [opSim; op(:,end)];
    end
    opinionsT = [opinionsT opSim];
end

%% Create surface plot
handle = figure;
xSize = size(opinionsT, 2); % Different Values of bb
HIST_BINS = 20; % Divisions of the [0,1] interval for the histogram
freqData = [];
for i = 1:xSize
    h = histogram(opinionsT(:,i), [0 1], 'Normalization', 'probability');
    h.NumBins = HIST_BINS; 
    freqData = [freqData; h.Values];
end

[X, Y] = meshgrid(1:xSize, 1:HIST_BINS);
surf(X,Y,freqData');
ax = gca;
%ax.XTick = 1:5:xSize;
ax.XTickLabel = logspace(2,-2, numel(ax.XTick));
ax.YTick = [1 5 10 15 20]; % Change this if you change HIST_BINS
ax.YTickLabel = linspace(0, 1, numel(ax.YTick));
%xlabel('Relative Size of B');
%staylabel('p');