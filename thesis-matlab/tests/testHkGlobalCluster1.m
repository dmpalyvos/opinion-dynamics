%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test if clustering appears when we change the å parameter of HK Global
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;

N = 625;
simMax = 50;
START_POINT = 0;
END_POINT = 0.4;
SAMPLES = 40;
STEP = 1;

opinionsT = [];
for opEps = linspace(START_POINT, END_POINT, SAMPLES)
    opSim = [];
    parfor i = 1:simMax
        s = rand(N,1);
        op = hkGlobal(s, opEps, 10e3, 1e-6);
        opSim = [opSim; op(:,end)];
    end
    opinionsT = [opinionsT opSim];
end

%% Calculate histograms
handle = figure;
xSize = size(opinionsT, 2); % Different Values of bb
HIST_BINS = 20; % Divisions of the [0,1] interval for the histogram
freqData = [];
for i = 1:xSize
    h = histogram(opinionsT(:,i), [0 1], 'Normalization', 'probability');
    h.NumBins = HIST_BINS; 
    freqData = [freqData; h.Values];
end

%% Plot
figure;
[X, Y] = meshgrid(1:xSize, 1:HIST_BINS);
surf(X,Y,freqData');
axis tight;
ax = gca;
ax.XTick = [1 10 20 30 40];
ax.XTickLabel = [0 0.1 0.2 0.3 0.4];
ax.YTick = [1 5 10 15 20]; % Change this if you change HIST_BINS
ax.YTickLabel = [0 0.25 0.5 0.75 1];
