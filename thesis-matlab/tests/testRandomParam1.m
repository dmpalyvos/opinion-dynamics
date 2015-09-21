close all;
clear all;

N = 50;
A = gnpConnected(N, 5/N);
A = A + diag(rand(size(A,1),1)*1e-3);
s = rand(N,1);

tMax = 50e3;
simMax = 32;
c = 1e-1;
%% Run
distances = [];
parfor i = 1:simMax
    fprintf(1,'i: %d\n',i);
    [equilibrium, op] = randomParametric(A, c, s, tMax, 1e-10);
    %[equilibrium, op] = randomParam(A, s, tMax, 1e-6);
    dist = bsxfun(@minus, op, equilibrium);
    dist = arrayfun(@(idx) norm(dist(:,idx), Inf), 1:tMax);
    distances = [distances; dist];
end
%% Plot
figure;
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C, 'EdgeColor', 'None');
x = 1:tMax;
y = mean(distances,1);
e = std(distances,1,1);
y1 = (y-e);
y2 = (y+e);

fill_between_lines(x,y1,y2,rgb('lightcoral'));
hold on;

plot(x,y, 'color', rgb('black'),'LineWidth', 0.01);
grid on;
xlabel('t');
axis([-inf,inf,0,inf]);