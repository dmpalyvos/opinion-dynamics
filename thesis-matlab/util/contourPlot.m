function [handle] = contourPlot(plotData, xlab, ylab, zlab)
global N START_POINT END_POINT STEP SAMPLES simMax MAX_P;

handle = figure;
xSize = size(plotData, 2); % K
ySize = size(plotData, 1); % p
[X, Y] = meshgrid(1:xSize, 1:ySize);
%surf(X,Y,plotData);
%surf(X,Y,plotData,'EdgeColor','interp');
contourf(X,Y,plotData,'EdgeColor','None');
%colormap summer;
axis tight;
ax = gca;
% X-Axis
xlabel(xlab);
JUMP_S = 30;
ax.XTick = 1:JUMP_S:SAMPLES;
ax.XTickLabel = START_POINT:STEP*JUMP_S:END_POINT;
% Y-Axis
ylabel(ylab);
ax.YTick = [1 20:20:simMax];
plist = 0:0.2:MAX_P;
%plist = linspace(0,MAX_P, numel(ax.YTick));
ax.YTickLabel = arrayfun(@(i) num2str(gnpDegree(N,plist(i)), '%2.0f'), 1:numel(plist), 'UniformOutput', false);
% Z-Axis
title(zlab);
%colorbar;
%title('HK K-NN Static');
end

