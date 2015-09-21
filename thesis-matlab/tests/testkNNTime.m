%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify that in static K-NN the convergence time increases dramatically
% just before the two large clusters merge into one
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
%clear all;

N = 128;
M = 100;

%A = gnpConnected(N, M/N);
% Intrinsic Beliefs
%s = rand(N,1);
timeData = [];
clusterData = []; % Number of clusters (more than 1/4 of total nodes)

HIST_BINS = 20; % Divisions of the [0,1] interval for the histogram
%% Run Simulation
gcp;
parfor K = 1:N
    [convergenceTime, finalOpinions] = kNN_NOMEM(A, s, K, 20e3, 1e-5); % Can simulate kNN2 too
    h = histogram(finalOpinions, [0 1], 'Normalization', 'probability');
    h.NumBins = HIST_BINS;
    timeData = [timeData convergenceTime];
    clusterData = [clusterData numel(find(h.Values > 0.2))]; % Number of clusters
end

%% Plot Number of clusters/Convergence Time
figure;
[ax, p1, p2] = plotyy(1:N, clusterData, 1:N, timeData);
title(sprintf('N = %d | Average Degree = %d', N, round(meanDegree(A))));
ylabel(ax(1),'#clusters') % label left y-axis
ylabel(ax(2),'Convergence time') % label right y-axis
xlabel('K');
ax(1).YTick = 0:max(clusterData);
ax(2).YTickMode = 'auto';