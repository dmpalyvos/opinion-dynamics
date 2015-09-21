function [handle] = surfacePlot(plotData, xlab, ylab, zlab)
global N START_POINT END_POINT STEP SAMPLES simMax MAX_P;

handle = figure;
xSize = size(plotData, 2); % K
ySize = size(plotData, 1); % p
[X, Y] = meshgrid(1:xSize, 1:ySize);
%surf(X,Y,plotData);
%surf(X,Y,plotData,'EdgeColor','none');
surfl(X,Y,plotData);
shading interp;
axis tight;
ax = gca;
% X-Axis
JUMP_S = 18;
ax.XTick = 1:JUMP_S:SAMPLES;
ax.XTickLabel = START_POINT:STEP*JUMP_S:END_POINT;
% Y-Axis
ax.YTick = [1 20:20:simMax];
plist = 0:0.2:MAX_P;
ax.YTickLabel = arrayfun(@(i) num2str(gnpDegree(N, plist(i)), '%2.0f'), 1:numel(plist), 'UniformOutput', false);
% Z-Axis
xlabel(xlab);
ylabel(ylab);
zlabel(zlab);
%title('HK K-NN Static');
end

