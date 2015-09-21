% Check if the K-NN2 model ALWAYS converges to an equilibrium
clear all;
close all;

P_NUM = 12; % Distinct p values
simMax = 4; % Repetitions on the same network with different opinions
T_MAX = 100e3;
convergenceTimes = [];

%% Gnp Networks
for N = 32:64:512
    parfor pIndex = 1:P_NUM
        pTable = linspace(0,1,P_NUM);
        p = pTable(pIndex);
        A = gnpConnected(N,p);
        for i = 1:simMax
            s = rand(N,1);
            [cT,~] = kNN_NOMEM(A, s, randi(N), T_MAX, 1e-6); % Can simulate kNN2 too
            convergenceTimes = [convergenceTimes cT];
        end
    end
end

%% Scale-Free Networks
for N = 32:64:512
    parfor m0Index = 1:P_NUM
        m0 = 3 + randi(N/2 - 3);
        A = scalefree(N,m0,1+randi(m0-2));
        for i = 1:simMax
            s = rand(N,1);
            [cT,~] = kNN_NOMEM(A, s, randi(N), T_MAX, 1e-5); % Can simulate kNN2 too
            convergenceTimes = [convergenceTimes cT];
        end
    end
end
%% Create histogram of Convergence Times
h = histogram(convergenceTimes,'Normalization','Probability');
h.NumBins = 50;
grid on;
h.FaceColor = rgb('Gray');
xlabel('Convergence Time');
