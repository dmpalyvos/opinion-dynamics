close all;
clear all;

global N START_POINT END_POINT STEP SAMPLES simMax MAX_P;

% Parameters
SAVE_GRAPH = false;

N = 128;
SIM_ID = 669;
CLASSIC_EPS = 0.04;

% Global Variables, should not change later
START_POINT = 2;
END_POINT = N;
STEP = 1;
SAMPLES = numel(START_POINT:STEP:END_POINT);
simMax = 100; 
MAX_P = 1; % Max value of p

OPINION_REPS = 1; % How many times to repeat for random opinions

distances = []; % Distance from Classic HK
convergenceTime = []; % Convergence Time
clustersNumber = []; % Number of clusters
%% Run Simulation
gcp;
% Intrinsic Beliefs
s = rand(N,1);
parfor z = 1:SAMPLES %K
    opDist = [];
    convT = [];
    clN = [];
    kTable = START_POINT:STEP:END_POINT;
    K = kTable(z);
    fprintf(1,'K: %d\n',K);
    for i = 1:simMax
        pTable = linspace(0, MAX_P, simMax);
        % Adjacency Matrix
        A = gnpConnected(N, pTable(i));
        avgD = 0; % Distance
        avgT = 0; % Time
        avgC = 0; % Clusters
        % Run
        for r = 1:OPINION_REPS
            [opHK] = hkGlobal(s, CLASSIC_EPS, 1e3, 1e-5);
            [cTime, finalOp] = kNN_NOMEM(A, s, K, 20e3, 1e-5); % Can simulate kNN2 too
            avgD = avgD + norm(opHK(:,end) - finalOp, 2)/OPINION_REPS;
            avgT = avgT + cTime/OPINION_REPS;
            % Clusters
            h = histogram(finalOp, [0 1], 'Normalization', 'probability');
            h.NumBins = 20;  
            avgC = avgC + numel(find(h.Values > 0.2))/OPINION_REPS;
        end
        opDist = [opDist; avgD];
        convT = [convT; avgT];
        clN = [clN; round(avgC)]
    end
    distances = [distances opDist];
    convergenceTime = [convergenceTime convT];
    clustersNumber = [clustersNumber clN];
end

%% Plot
surfacePlot(convergenceTime,'','',''); colormap(flipud(gray));
surfacePlot(clustersNumber,'','','');colormap(gray);

%% Save Data
save([num2str(SIM_ID) '_data.mat']);

