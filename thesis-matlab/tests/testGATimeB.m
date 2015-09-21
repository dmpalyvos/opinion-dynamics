close all;
clear all;

N = 128;
simMax = 25;
SAMPLES = 28;
START_POINT = -1;
END_POINT = 1;
tTable = []; % Convergence Time
qTable = []; % Q Average Max Eigenvalue
P_VAL = 1e-1;
parfor k = 1:SAMPLES
    bI = logspace(START_POINT,END_POINT,SAMPLES);
    bb = bI(k);
    tMax = 0;
    qEig = 0;
    for i = 1:simMax
        % Intrinsic beliefs
        s = rand(N,1);
        % Stubborness
        sB = bb*rand(N,1);
        % Adjacency Matrix
        A = gnpConnected(N, P_VAL);
        % Run
        [eigQ, op] = generalizedAsymmetric(A, sB, s, 10e3, 1e-6);
        tMax = tMax + (size(op, 2)/simMax);
        qEig = qEig + eigQ/simMax;
    end
    tTable = [tTable tMax];
    qTable = [qTable qEig];
end

%% Plot Convergence Time
figure;
xVal = logspace(START_POINT,END_POINT,SAMPLES);
semilogx(xVal, tTable,'k');
ylabel('Convergence Time') % label left y-axis
xlabel('Relative size of B');
grid on;

%% Plot Eigenvalue
figure;
xVal = logspace(START_POINT,END_POINT,SAMPLES);
semilogx(xVal, qTable,'k');
ylabel('\lambda _{max}') 
xlabel('Relative size of B');
ax = gca;
axis([-inf,inf,0,1]);
grid on;
%legend('Convergence Time (Degree = 2)','Convergence Time (Degree = 6)','Convergence Time (Degree = 40)','Eigenvalue (Degree = 2)','Eigenvalue (Degree = 6)','Eigenvalue (Degree = 40)');