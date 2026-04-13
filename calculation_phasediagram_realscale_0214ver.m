clear; clc;

% ===== INPUT =====
csvFile = "phase_timeseries_redsweep_0326_10frame_50.csv";
T = readtable(csvFile);
p = T.p_red;
v = T.mean_speed;
x = T.corr_len;
CMap=slanCM('iceburn');
% ===== 옵션 =====
useTop10PerSample = false;   % 샘플당 top10만 쓰기
topK              = 10;
% 배경 표현 선택
bgMode = "scatter";         % "scatter" 또는 "density" 또는 "none"
bgAlpha = 0.1;             % scatter 투명도 (0.03~0.15 추천)
densityBins = 30;           % density 모드 bin

% 에러바/요약
useSEM         = false;
errVisualScale = 1.0;        % 시각적으로 에러바 줄이기(0~1)
useCaps        = true;
capFrac        = 0.012;      % cap 길이 (축 범위 대비)

markerSizeMean = 80;
markerSizeBg   = 50;
% ===== 1) TopK per sample 선택 =====
if useTop10PerSample
    if all(ismember(["seed","rep"], T.Properties.VariableNames))
        key = strcat("p",string(p),"_s",string(T.seed),"_r",string(T.rep));
    elseif ismember("sample_id", T.Properties.VariableNames)
        key = string(T.sample_id);
    else
        key = strcat("p",string(p));
        warning("seed/rep/sample_id 없음 → p_red별 topK만 선택됨(샘플별 topK 아님).");
    end
    v0 = (v - min(v)) / max(eps,(max(v)-min(v)));
    x0 = (x - min(x)) / max(eps,(max(x)-min(x)));
    score = x0;

    keep = false(height(T),1);
    [uk,~,g] = unique(key);
    for gi = 1:numel(uk)
        idx = (g==gi);
        [~,ord] = sort(score(idx),'descend');
        ord = ord(1:min(topK,numel(ord)));
        ii = find(idx);
        keep(ii(ord)) = true;
    end

    p = p(keep); v = v(keep); x = x(keep);
end
% ===== 1) Median per sample =====

if all(ismember(["seed","rep"], T.Properties.VariableNames))
    key = strcat("p",string(T.p_red),"_s",string(T.seed),"_r",string(T.rep));
elseif ismember("sample_id", T.Properties.VariableNames)
    key = string(T.sample_id);
else
    error("sample identifier (seed/rep or sample_id) needed");
end

[uk,~,g] = unique(key);

p_med = zeros(numel(uk),1);
v_med = zeros(numel(uk),1);
x_med = zeros(numel(uk),1);

for i = 1:numel(uk)

    idx = (g==i);

    p_med(i) = median(T.p_red(idx),"omitnan");
    v_med(i) = mean(T.mean_speed(idx),"omitnan");
    x_med(i) = mean(T.corr_len(idx),"omitnan");

end

p = p_med;
v = v_med;
x = x_med;
% ===== 2) Normalize using specific reference p_red values =====
% --- Reference for speed: p_red = 0 ---
idx_v_ref = abs(p - 0) < 1e-12;
V_ref = mean(v(idx_v_ref), "omitnan");

V_ref = 1;

% --- Reference for correlation length: p_red = 1 ---
idx_x_ref = abs(p - 1) < 1e-12;
X_ref = mean(x(idx_x_ref), "omitnan");

X_ref = 2;

% Safety check
if isempty(V_ref) || isnan(V_ref)
    error('No kept samples found for p_red = 0 (speed reference)');
end
if isempty(X_ref) || isnan(X_ref)
    error('No kept samples found for p_red = 1 (corr length reference)');
end

% Normalize
vN = v / V_ref;
xN = x / X_ref;


% ===== 3) p_red별 요약(mean±SEM/STD) =====
p_list = sort(unique(p(:)))';
nP = numel(p_list);

mV = nan(nP,1); eV = nan(nP,1);
mX = nan(nP,1); eX = nan(nP,1);
nN = nan(nP,1);

for i = 1:nP
    pr = p_list(i);
    idx = abs(p - pr) < 1e-12;

    vv = vN(idx);  xx = xN(idx);
    nN(i) = sum(idx);

    mV(i) = median(vv, "omitnan");
    mX(i) = median(xx, "omitnan");

    sV = std(vv, "omitnan");
    sX = std(xx, "omitnan");

    if useSEM
        eV(i) = sV / sqrt(max(nN(i),1));
        eX(i) = sX / sqrt(max(nN(i),1));
    else
        eV(i) = sV;
        eX(i) = sX;
    end
end

% ===== 4) 축 바꾸기: X=Correlation, Y=Speed =====
Xmean = mX;  Ymean = mV;
Xerr  = eX;  Yerr  = eV;

Xbg = xN;    Ybg = vN;

% ===== 5) 컬러(자연스러운 연속) =====
cmap = CMap;
cval = (p_list - min(p_list)) / max(eps, (max(p_list)-min(p_list)));
cidx = max(1, min(256, round(1 + cval*255)));
cols = cmap(cidx,:);

% bg scatter용: 각 점의 p값도 색으로
pNorm = (p - min(p_list)) / max(eps, (max(p_list)-min(p_list)));
pIdx  = max(1, min(256, round(1 + pNorm*255)));
bgCol = cmap(pIdx,:);

% ===== 6) Plot =====
figure('Color','w'); hold on; box on;
set(gca,'LineWidth',1.2,'FontSize',12,'TickDir','out');
grid off;

xlabel("Normalized correlation length  \xi/<\xi>_{all}");
ylabel("Normalized speed  <v>/<v>_{all}");
title("Phase diagram by p_{red} (background + mean±error)");

% 기준선 (1,1)
xline(0.5,'k--','LineWidth',1.0,'Alpha',0.30);
yline(0.75,'k--','LineWidth',1.0,'Alpha',0.30);

% ----- (A) Background: scatter or density -----
switch bgMode
    case "scatter"
        % 뒤에 원 데이터 점들(색=p_red, 투명)
        scatter(Xbg, Ybg, markerSizeBg, bgCol, 'filled', ...
            'MarkerFaceAlpha', bgAlpha, 'MarkerEdgeAlpha', 0);

    case "density"
        % 2D 밀도맵 (p_red 무시하고 전체 분포만)
        xedges = linspace(min(Xbg), max(Xbg), densityBins+1);
        yedges = linspace(min(Ybg), max(Ybg), densityBins+1);
        H = histcounts2(Xbg, Ybg, xedges, yedges);

        % 보기 좋게 log 스케일(선택)
        H = log1p(H);

        % imagesc는 축이 뒤집히기 쉬워서 transpose + axis xy
        imagesc(0.5*(xedges(1:end-1)+xedges(2:end)), ...
                0.5*(yedges(1:end-1)+yedges(2:end)), ...
                H');
        axis xy;
        colormap(gca, "gray");    % 배경은 중립
        set(gca,'CLim',[min(H(:)) max(H(:))]);
        alpha(0.3);              % 배경 맵 투명도

    otherwise
        % none
end

% ----- (B) mean trajectory line -----
plot(Xmean, Ymean, '--', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5);

% cap 길이
xr = range(Xmean); if xr==0, xr = 1; end
yr = range(Ymean); if yr==0, yr = 1; end
capX = capFrac * xr;
capY = capFrac * yr;

% ----- (C) mean points + errorbars with caps -----
for i = 1:nP
    ex = errVisualScale * Xerr(i);
    ey = errVisualScale * Yerr(i);
    col = cols(i,:);

    % error bars
    plot([Xmean(i)-ex, Xmean(i)+ex], [Ymean(i), Ymean(i)], '-', 'Color', [col 0.7], 'LineWidth', 1.6);
    plot([Xmean(i), Xmean(i)], [Ymean(i)-ey, Ymean(i)+ey], '-', 'Color', [col 0.7], 'LineWidth', 1.6);

    if useCaps
        % horizontal caps
        plot([Xmean(i)-ex, Xmean(i)-ex], [Ymean(i)-capY, Ymean(i)+capY], '-', 'Color', [col 0.7], 'LineWidth', 1.6);
        plot([Xmean(i)+ex, Xmean(i)+ex], [Ymean(i)-capY, Ymean(i)+capY], '-', 'Color', [col 0.7], 'LineWidth', 1.6);
        % vertical caps
        plot([Xmean(i)-capX, Xmean(i)+capX], [Ymean(i)-ey, Ymean(i)-ey], '-', 'Color', [col 0.7], 'LineWidth', 1.6);
        plot([Xmean(i)-capX, Xmean(i)+capX], [Ymean(i)+ey, Ymean(i)+ey], '-', 'Color', [col 0.7], 'LineWidth', 1.6);
    end

    scatter(Xmean(i), Ymean(i), markerSizeMean, col, 'filled', ...
        'MarkerFaceAlpha', 0.95, 'MarkerEdgeColor', [0 0 0], 'MarkerEdgeAlpha', 0.4);

    % text(Xmean(i), Ymean(i), sprintf("  p=%.1f", p_list(i)), ...
    %     'Color', [0.15 0.15 0.15], 'FontSize', 11, 'VerticalAlignment','middle');
end
% ===== grid for KDE =====
nx = 120;
ny = 120;

xgrid = linspace(min(xN), max(xN), nx);
ygrid = linspace(min(vN), max(vN), ny);

[Xg,Yg] = meshgrid(xgrid,ygrid);

figure('Color','w'); hold on; box on;
set(gca,'LineWidth',1.2,'FontSize',12,'TickDir','out');

xlabel("Correlation length  \xi / 2");
ylabel("Speed  v / 1");
title("Phase diagram (KDE by p_{red})");

colormap(CMap);

for i = 1:nP

    pr = p_list(i);
    idx = abs(p - pr) < 1e-12;

    xi = xN(idx);
    yi = vN(idx);

    if numel(xi) < 5
        continue
    end

    % ----- 2D KDE -----
    [f,~] = ksdensity([xi yi],[Xg(:) Yg(:)]);
    F = reshape(f,size(Xg));

    % normalize density (visualization)
    F = F / max(F(:));

    % ----- contour plot -----
    contour(Xg,Yg,F,[0.2 0.4 0.6 0.8],...
        'LineWidth',1.5,...
        'Color',cols(i,:));

end

% mean trajectory line
plot(Xmean,Ymean,'--','Color',[0.2 0.2 0.2],'LineWidth',1.5);

scatter(Xmean,Ymean,80,cols,'filled','MarkerEdgeColor','k')

xlim([0.2 1.5]);
ylim([0.6 1.3]);

% ----- colorbar (p_red) -----
if bgMode == "scatter" || true
    colormap(CMap);
    cb = colorbar;
    cb.Label.String = "p_{red}";
    caxis([min(p_list) max(p_list)]);
end

% axis padding
xlim([2.0, 10.0]);
ylim([0.0, 1.0]);

function cmap = greenBlackRed(n)
% Publication-safe Green–Black–Red colormap
% n: number of colors (e.g., 256)

if nargin < 1
    n = 256;
end

% Dark green → black
g1 = [0.0, 0.6, 0.25];   % dark green (safe)
mid = [0.8, 0.8, 0.8];   % black
r2 = [0.9, 0.0, 0.10]; % dark red (safe)

n1 = floor(n/2);
n2 = n - n1;

c1 = [linspace(g1(1), mid(1), n1)', ...
      linspace(g1(2), mid(2), n1)', ...
      linspace(g1(3), mid(3), n1)'];

c2 = [linspace(mid(1), r2(1), n2)', ...
      linspace(mid(2), r2(2), n2)', ...
      linspace(mid(3), r2(3), n2)'];

cmap = [c1; c2];
end