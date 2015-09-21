close all;
clear all;

N = 128; % Number of nodes

fprintf(1, 'Starting simulation with %d nodes\n', N);

s = rand(N,1);
sB = rand(N,1);

simMax = 25;
SAMPLES = 28;
START_POINT = 0;
END_POINT = 1;

tTable = []; % Convergence Time
qTable = []; % Q Average Max Eigenvalue
parfor pI = 1:SAMPLES
    pChoice = linspace(START_POINT, END_POINT, SAMPLES);
    p = pChoice(pI);
    tMax = 0;
    qEig = 0;
    aEig = 0;
    for i = 1:simMax
        A = gnpConnected(N, p);
        [eigQ, opinions] = generalizedAsymmetric(A, sB, s, 5e3, 1e-10);
        tMax = tMax + (size(opinions, 2)/simMax);
        qEig = qEig + eigQ/simMax;
    end
    tTable = [tTable tMax];
    qTable = [qTable qEig];
end
%% Plot
handle = figure;
pTable = linspace(START_POINT, END_POINT, SAMPLES);
%[ax, p1, p2] = plotyy(pTable, tTable, pTable, qTable);
plot(pTable,qTable,'k');
pList = linspace(START_POINT, END_POINT, 5);
ax = gca;
ax.XTick = pList;
xlabel('p') % label x-axis
ylabel('\lambda _{max}')
grid on;
