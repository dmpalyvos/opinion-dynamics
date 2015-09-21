close all;
clear all;

N = 128;

s = rand(N,1);
sB = rand(N,1);

simMax = 25;
SAMPLES = 28;
START_POINT = 0;
END_POINT = 1;

tTable = [];
degreeTable = [];
pTable = [];
parfor pI = 1:SAMPLES
    pChoice = linspace(START_POINT, END_POINT, SAMPLES);
    p = pChoice(pI);
    tMax = 0;
    tDeg = 0;
    for i = 1:simMax
        A = gnpConnected(N, p);
        [~, opinions] = generalizedAsymmetric(A, sB, s, 5e3, 1e-10);
        tMax = tMax + (size(opinions, 2)/simMax);
        tDeg = tDeg + (meanDegree(A)/simMax);
    end
    tTable = [tTable tMax];
    degreeTable = [degreeTable tDeg];
    pTable = [pTable p];
end

%% Plot
handle = figure;
[ax, p1, p2] = plotyy(pTable, tTable, pTable, degreeTable);
ylabel(ax(1),'Convergence time') % label left y-axis
ylabel(ax(2),'Average Degree') % label right y-axis
axis(ax(1), [-inf inf min(tTable)-10 max(tTable)+10]);
axis(ax(2), [-inf, inf, 2, N])
set(ax(1), 'YTickMode', 'auto');
set(ax(2), 'YTickMode', 'auto');
xlabel('p') % label x-axis

