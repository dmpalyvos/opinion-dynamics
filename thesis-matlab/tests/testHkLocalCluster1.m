close all;
clear all;

N = 625;
simMax = 50;
START_POINT = 0;
END_POINT = 0.4;
SAMPLES = 40;
opinionsT = [];
eTable = linspace(START_POINT, END_POINT, SAMPLES);
parfor k = 1:SAMPLES
    opEps = eTable(k);
    fprintf(1,'å: %3.2f\n', opEps);
    opSim = [];
    for i = 1:simMax
        pTable = linspace(0, 0.01,simMax); % Network Degree Range
        s = rand(N,1);
        A = gnpConnected(N, pTable(i));
        [~, op] = hkLocal(A, s, opEps, 10e3, 1e-6);
        opSim = [opSim; op(:,end)];
    end
    opinionsT = [opinionsT opSim];
end
%% Create surface plot
handle = figure;
xSize = size(opinionsT, 2); % Different Values 
HIST_BINS = 20; % Divisions of the [0,1] interval for the histogram
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
ax.XTick = [1 10 20 30 40];
ax.XTickLabel = [0 0.1 0.2 0.3 0.4];
ax.YTick = [1 5 10 15 20]; % Change this if you change HIST_BINS
ax.YTickLabel = [0 0.25 0.5 0.75 1];
