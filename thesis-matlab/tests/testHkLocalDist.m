%% HK Local distance from HK Global

close all;
clear all;

global N START_POINT END_POINT STEP SAMPLES simMax MAX_P;

% Parameters
SAVE_GRAPH = false;

N = 128;
SIM_ID = 84847;
% Global Variables, should not change later
START_POINT = 0;
END_POINT = 0.4;
STEP = 0.01;
SAMPLES = numel(START_POINT:STEP:END_POINT);
simMax = 100; 
MAX_P = 1; % Max value of p

OPINION_REPS = 16; % How many times to repeat for random opinions

distances = []; % Distance from Classic HK
convergenceTime = []; % Convergence Time
clustersNumber = []; % Number of clusters
%% Run Simulation
gcp;
% Intrinsic Beliefs
parfor z = 1:SAMPLES %K
    opDist = [];
    convT = [];
    clN = [];
    eTable = linspace(START_POINT, END_POINT, SAMPLES);
    myEps = eTable(z);
    fprintf(1,'å: %d\n',myEps);
    for i = 1:simMax
        pTable = linspace(0, MAX_P, simMax);
        % Adjacency Matrix
        A = gnpConnected(N, pTable(i));
        avgD = 0; % Distance
        avgT = 0; % Time
        avgC = 0; % Clusters
        % Run
        for r = 1:OPINION_REPS
            s = rand(N,1);
            [opHK] = hkGlobal(s, myEps, 1e3, 1e-5);
            [~, op1] = hkLocal(A, s, myEps, 10e3, 1e-5);
            avgD = avgD + norm(opHK(:,end) - op1(:,end), 2)/OPINION_REPS;
            avgT = avgT + size(op1,2)/OPINION_REPS;
            % Clusters
            h = histogram(op1(:,end), [0 1], 'Normalization', 'probability');
            h.NumBins = 40;  
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
surfacePlot(distances,'', '', ''); %colormap(gray); 

%% Save Data
save([num2str(SIM_ID) '_data.mat']);
