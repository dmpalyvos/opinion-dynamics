close all;
clear all;

N = 32; 
s = rand(N,1);

tMax = 10e6;
SIM_REPS = 32;
SIM_ID = input('ID: ');
%% Run 
times = [];
errors = [];
normList = [];
mTable = [10 4 2 1 0.7 0.4 0.2 0.15 0.1]; % Multipliers for diagonal
gcp;
for N_MUL = mTable
    % Weights/Adjacency Matrix
    A = gnpConnected(N, 5/N);
    % Split the weight in half between the diagonal and the other nodes
    A = A - diag(diag(A));
    A = A + diag(sum(A,2)*N_MUL);
    A = normalizeMatrix(A);
    fprintf(1,'||A|| = %4.3f\n',norm(A-diag(diag(A)),Inf));
    normList = [normList norm(A-diag(diag(A)),Inf)];
    tt = [];
    parfor i = 1:SIM_REPS
        fprintf(1,'i: %d\n',i);
        % Ensure that the diagonal is not zero
        %A = A + diag(rand(size(A,1),1) * 1e-3);
        % Run
        tM = random_NOMEM(A, s, tMax, 1e-3); % Convergence Time
        tt = [tt; tM]; 
    end
    times = [times mean(tt)];
    errors = [errors std(tt)];
end
%% Plot 
figure;
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C, 'EdgeColor', 'None');
x = 1:numel(mTable);
%y = mean(distances,1);
%e = std(distances,1,1);
y = times;
e = errors;
y1 = (y-e);
y2 = (y+e);
% plot(x,y+e,'r:');
% plot(x, y-e,'r:');
fill_between_lines(x,y1,y2,rgb('lightcoral'));
hold on;
plot(x,y, 'color', rgb('black'));
%title('Convergence Time');
ylabel('Rounds') % y-axis
xlabel('||A||_{\infty}');
ax = gca;
ax.XTick = 1:numel(mTable);
ax.XTickLabel = arrayfun(@(i) num2str(normList(i),'%3.2f'), 1:numel(mTable),'UniformOutput', false);
axis([-inf,inf,0,tMax]);

save([num2str(SIM_ID) '_data.mat']);
