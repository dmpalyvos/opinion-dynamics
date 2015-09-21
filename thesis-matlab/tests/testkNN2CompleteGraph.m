% Time needed for the K-NN2 model to create a complete graph
close all;
clear all;

global N START_POINT END_POINT STEP SAMPLES simMax MAX_P;

% Global Variables, should not change later
N = 128;
START_POINT = 2;
END_POINT = N;
STEP = 1;
SAMPLES = numel(START_POINT:STEP:END_POINT);
simMax = 100; 
MAX_P = 1; % Max value of p

OPINION_REPS = 1; % How many times to repeat for random opinions

timeFullDeg = []; % Number of rounds to create full network
%% Run Simulation
gcp;
% Intrinsic Beliefs
s = rand(N,1);
parfor z = 1:SAMPLES %K
    clN = [];
    kTable = START_POINT:STEP:END_POINT;
    K = kTable(z);
    fprintf(1,'K: %d\n',K);
    for i = 1:simMax
        pTable = linspace(0, MAX_P, simMax);
        A = gnpConnected(N, pTable(i));
        % Run
        [~,~,~,cT] = kNN2(A,s,K,1e3,1e-5);
        %lastChange = find(diff(deg),1,'last');
        %if (isempty(lastChange))
        %    lastChange = 0;
        %end
        clN = [clN; cT]
    end
    timeFullDeg = [timeFullDeg clN];
end

%% Plot
contourPlot(timeFullDeg,'K','Degree',''); caxis([0 2]); 
