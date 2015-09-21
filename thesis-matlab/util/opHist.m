function values = opHist(opinions, t)
%OPHIST Plot histogram of the opinion vector at some t
if (t == 0)
    t = size(opinions,2);
end
figure;
h = histogram(opinions(:,t), [0 1]);
h.NumBins = 100;
h.FaceColor = rgb('DarkGray');
values = h.Values;
ylabel('#Nodes');
xlabel('Opinion');
grid on;
%title(['t = ' num2str(t)]);
end

