% Two-species collective motion in circular confinement
% - RR: stick (perfectly inelastic) when approaching & within r_coll
%       Contact persists until pair distance > r_reset (hysteresis)
%       While contact persists, enforce cluster COM velocity (DSU)
% - GG: (near) elastic collision on normal component (optional)
% - RG: strong active repulsive impulse (nonconservative optional)
% - Pair cooldown to avoid repeated collision-trigger while still close
% - Narrow weak Vicsek-B (always-on), optional apply to only Red
% - Fast neighbor search via grid binning (grid_pairs_fast)

clear; clc; rng(0);

%% ===================== USER SETTINGS =====================
% ===================== IMAGE EXPORT =====================
save_frames = true;
img_dir = 'frames_300dpi_327_new_075';
img_format = 'png';   % 'png' or 'tif'
dpi = 300;

if save_frames
    if ~exist(img_dir,'dir')
        mkdir(img_dir);
    end
end

% Domain
R_domain = 30;         % radius
k_wall   = 50.0;        % soft wall accel strength

% Simulation
N      = 600;
p_red  = 0.75;          % fraction red (set 1.0 for all red)
dt     = 0.05;
Tsteps = 2000;

% --- Persistent Gaussian noise (OU) state ---
use_OU_noise = true;
tau_R = 1.0;      % persistence time (in time units)
tau_G = 1.0;
sigma_R = 0.7;    % OU accel std scale
sigma_G = 0.7;

ou = zeros(N,2);  % OU acceleration state

% Speeds
v_init = 1.0;
vmax   = 2.5;

% Drag (type-dependent)
gamma_R = 0.0;
gamma_G = 0.0;

% Noise (accel-like, type-dependent)
eta_R = 0.7;
eta_G = 0.7;

% Alignment (RR Vicsek-like accel OR use Vicsek-B below)
% (keep J_align = 0 if you only want collision-driven + Vicsek-B)
r_align = 2.0;
J_align = 0.00;         % set small (e.g., 0.3) if you want mild RR directional bias

% % Collision radii
r_coll = 2.0;
% reset_gap_factor = 1.10;        % hysteresis: contact removed only if d > r_reset
% r_reset = reset_gap_factor * r_coll;

% Cooldown for same pair collision-trigger (steps)
cooldown_steps = 20;             % 5~15 usually good

% Mass
mR = 1.0;
mG = 1.0;

% GG restitution (elasticity)
enable_GG_collision = true;
e_GG = 1.0;                    % 1=elastic, 0=inelastic

% RG collision baseline restitution-like
e_RG = 1.0;

% Overlap correction (position)
pos_correction = 2.0;           % 0~1.0; lower = gentler
eps_sep = 0.001;                 % small extra separation to avoid re-trigger jitter

% Optional: approaching condition for RR stick start
RR_require_approach = true;

% Vicsek-B (always-on, narrow & weak)
use_vicsek_B = true;

% --- type-specific Vicsek-B ---
r_vicsek_R   = 2.0;
alpha_v_R    = 0.1;
p_vicsek_R   = 1.0;     % 10% percentage

r_vicsek_G   = 2.0;
alpha_v_G    = 0.1;
p_vicsek_G   = 1.0;

vicsek_warmup_R = 0;
vicsek_warmup_G = 0;  

% Visualization / video
plot_every = 8;
% make_video = true;
% video_name = 'two_species_paircool_full_fast_0.mp4';

%% ===================== INIT =====================

Nr = round(N*p_red);
Ng = N-Nr;

types = [ones(Nr,1); zeros(Ng,1)];   % 1=Red, 0=Green
isR = (types==1);
isG = ~isR;

% Positions uniform in disk
theta = 2*pi*rand(N,1);
rad   = R_domain * sqrt(rand(N,1));
pos   = [rad.*cos(theta), rad.*sin(theta)];

% Velocities random direction
dir0 = 2*pi*rand(N,1);
vel  = v_init * [cos(dir0), sin(dir0)];

% Contact memory + cooldown
% in_contact = false(N,N);             % RR-only meaningful, but keep generic
cooldown   = zeros(N,N,'uint16');    % steps remaining; symmetric maintained

% Metrics
pol_red    = nan(Tsteps,1);
pol_green  = nan(Tsteps,1);
pol_all    = nan(Tsteps,1);
mean_speed = nan(Tsteps,1);

coll_RR = nan(Tsteps,1);
coll_GG = nan(Tsteps,1);
coll_RG = nan(Tsteps,1);

figure('Color','w');
ax = axes('Position',[0.05 0.05 0.90 0.90]);  
axis(ax,'equal'); box(ax,'on'); hold(ax,'on');
set(ax,'XTick',[],'YTick',[]);

% if make_video
%     vwriter = VideoWriter(video_name,'MPEG-4');
%     vwriter.FrameRate = max(10, round(1/dt));
%     open(vwriter);
% end

%% ===================== MAIN LOOP =====================
for t = 1:Tsteps

    % --- cooldown decrement (sparse update would be faster, but OK for N~1000) ---
    if any(cooldown(:) > 0)
        cooldown(cooldown>0) = cooldown(cooldown>0) - 1;
    end

    % ---------- (A) RR alignment accel (optional) ----------
    spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
    u   = vel ./ spd;
    align_acc = zeros(N,2);

    if J_align > 0 && Nr > 1
        pairsA = grid_pairs_fast(pos, r_align, R_domain);
        if ~isempty(pairsA)
            sumUx = zeros(N,1); sumUy = zeros(N,1); cnt = zeros(N,1);
            i = pairsA(:,1); j = pairsA(:,2);
            RRmask = isR(i) & isR(j);
            iRR = i(RRmask); jRR = j(RRmask);

            sumUx(iRR) = sumUx(iRR) + u(jRR,1); sumUy(iRR) = sumUy(iRR) + u(jRR,2); cnt(iRR) = cnt(iRR) + 1;
            sumUx(jRR) = sumUx(jRR) + u(iRR,1); sumUy(jRR) = sumUy(jRR) + u(iRR,2); cnt(jRR) = cnt(jRR) + 1;

            idx = isR & (cnt>0);
            meanDir = [sumUx(idx)./cnt(idx), sumUy(idx)./cnt(idx)];
            align_acc(idx,:) = J_align * meanDir;
        end
    end

    % ---------- (B) wall ----------
    rnorm = hypot(pos(:,1),pos(:,2));
    over  = max(0, rnorm - R_domain);
    wall_dir = -[pos(:,1)./max(rnorm,1e-9), pos(:,2)./max(rnorm,1e-9)];
    wall_acc = (k_wall * over) .* wall_dir;
    
    % ---------- (C) persistent Gaussian noise (OU) ----------
    if use_OU_noise
        tau = tau_R*ones(N,1);   tau(isG) = tau_G;
        sig = sigma_R*ones(N,1); sig(isG) = sigma_G;
    
        a = exp(-dt ./ max(tau,1e-9));   % per-particle decay
        % exact OU discretization
        ou = ou .* a + (sig .* sqrt(1 - a.^2)) .* randn(N,2);
    
        noise_acc = ou;   % use OU state as acceleration noise
    else
        eta = P.eta_R*ones(N,1); eta(isG) = P.eta_G;
        noise_acc = eta .* randn(N,2);
    end

    % ---------- (D) drag ----------
    gamma = gamma_R*ones(N,1); gamma(isG) = gamma_G;
    drag_acc = -gamma .* vel;

    % ---------- (E) integrate free step ----------
    acc = align_acc + wall_acc + noise_acc + drag_acc;

    % optional accel cap for stability
    max_acc = 12.0;
    an = max(hypot(acc(:,1),acc(:,2)), 1e-12);
    acc = acc .* min(1, max_acc./an);

    vel = vel + dt * acc;

    % speed cap
    spd = hypot(vel(:,1),vel(:,2));
    tooFast = spd > vmax;
    if any(tooFast)
        vel(tooFast,:) = vel(tooFast,:) .* (vmax ./ spd(tooFast));
    end

    pos = pos + dt * vel;

    % mild outside correction
    rnorm = hypot(pos(:,1),pos(:,2));
    outside = rnorm > (R_domain + 0.5);
    if any(outside)
        corr_dir = -[pos(outside,1)./rnorm(outside), pos(outside,2)./rnorm(outside)];
        pos(outside,:) = pos(outside,:) + 0.2 * corr_dir;
    end

    % ---------- (F) Vicsek-B (type-specific) ----------
    if use_vicsek_B

        % (1) RED Vicsek
        if t > vicsek_warmup_R && alpha_v_R > 0 && any(isR)
            pairsV = grid_pairs_fast(pos, r_vicsek_R, R_domain);
            if ~isempty(pairsV)
                spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
                u   = vel ./ spd;

                sumUx = zeros(N,1); sumUy = zeros(N,1); cnt = zeros(N,1);
                i = pairsV(:,1); j = pairsV(:,2);

                sumUx(i) = sumUx(i) + u(j,1); sumUy(i) = sumUy(i) + u(j,2); cnt(i) = cnt(i) + 1;
                sumUx(j) = sumUx(j) + u(i,1); sumUy(j) = sumUy(j) + u(i,2); cnt(j) = cnt(j) + 1;

                idx = isR & (cnt>0);

                % apply probability (optional)
                if p_vicsek_R < 1
                    idx = idx & (rand(N,1) < p_vicsek_R);
                end

                if any(idx)
                    md = [sumUx(idx)./cnt(idx), sumUy(idx)./cnt(idx)];
                    md = md ./ max(hypot(md(:,1),md(:,2)), 1e-12);

                    vmag = spd(idx);
                    v_align = md .* vmag;

                    vel(idx,:) = (1-alpha_v_R)*vel(idx,:) + alpha_v_R*v_align;
                end
            end
        end

        % (2) GREEN Vicsek
        if t > vicsek_warmup_G && alpha_v_G > 0 && any(isG)
            pairsV = grid_pairs_fast(pos, r_vicsek_G, R_domain);
            if ~isempty(pairsV)
                spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
                u   = vel ./ spd;

                sumUx = zeros(N,1); sumUy = zeros(N,1); cnt = zeros(N,1);
                i = pairsV(:,1); j = pairsV(:,2);

                sumUx(i) = sumUx(i) + u(j,1); sumUy(i) = sumUy(i) + u(j,2); cnt(i) = cnt(i) + 1;
                sumUx(j) = sumUx(j) + u(i,1); sumUy(j) = sumUy(j) + u(i,2); cnt(j) = cnt(j) + 1;

                idx = isG & (cnt>0);

                % apply probability (optional)
                if p_vicsek_G < 1
                    idx = idx & (rand(N,1) < p_vicsek_G);
                end

                if any(idx)
                    md = [sumUx(idx)./cnt(idx), sumUy(idx)./cnt(idx)];
                    md = md ./ max(hypot(md(:,1),md(:,2)), 1e-12);

                    vmag = spd(idx);
                    v_align = md .* vmag;

                    vel(idx,:) = (1-alpha_v_G)*vel(idx,:) + alpha_v_G*v_align;
                end
            end
        end

    end
% ---------- (G) collisions: ONE-SHOT RR inelastic event + cooldown ----------
pairsC = grid_pairs_fast(pos, r_coll, R_domain);

J_RR = 0; J_GG = 0; J_RG = 0;

if ~isempty(pairsC)
    for kk = 1:size(pairsC,1)
        i = pairsC(kk,1);
        j = pairsC(kk,2);

        dx = pos(i,1) - pos(j,1);
        dy = pos(i,2) - pos(j,2);
        d  = hypot(dx,dy);

        if d <= 1e-12 || d >= r_coll
            continue;
        end

        n = [dx, dy] / d;          % normal from j->i
        overlap = r_coll - d;

        % --- position overlap correction (soft) ---
        if overlap > 0
            corr = 0.5 * pos_correction * (overlap + eps_sep) * n;
            pos(i,:) = pos(i,:) + corr;
            pos(j,:) = pos(j,:) - corr;
        end

        iR = isR(i); jR = isR(j);
        iG = ~iR;    jG = ~jR;
        isRGpair = (iR && jG) || (iG && jR);

        if isRGpair && cooldown(i,j) > 0
            continue;
        end

        % ---------- RR: one-shot perfectly inelastic ----------
        if iR && jR
            % approach condition (recommended)
            vrel = vel(i,:) - vel(j,:);
            vn = dot(vrel, n);   % >0 means separating along n (i moving away from j)
            if RR_require_approach && vn >= 0
                continue;
            end

            mi = mR; mj = mR;
            vcm = (mi*vel(i,:) + mj*vel(j,:)) / (mi + mj);

            dv_i = vcm - vel(i,:);
            dv_j = vcm - vel(j,:);

            vel(i,:) = vcm;
            vel(j,:) = vcm;

            % collision intensity proxy
            J_RR = J_RR + (mi*norm(dv_i) + mj*norm(dv_j));

            % set cooldown
            cooldown(i,j) = uint16(cooldown_steps);
            cooldown(j,i) = uint16(cooldown_steps);

        % ---------- GG: elastic ----------
        elseif iG && jG
            if ~enable_GG_collision
                continue;
            end

            mi = mG; mj = mG;
            vrel = vel(i,:) - vel(j,:);
            vn = dot(vrel, n);
            if vn < 0
                j_imp = -(1+e_GG) * vn / (1/mi + 1/mj);
                Jvec  = j_imp * n;
                vel(i,:) = vel(i,:) + (Jvec/mi);
                vel(j,:) = vel(j,:) - (Jvec/mj);
                J_GG = J_GG + norm(Jvec);

                cooldown(i,j) = uint16(cooldown_steps);
                cooldown(j,i) = uint16(cooldown_steps);
            end

            % ---------- RG: treat same as GG elastic ----------
        else
            mi = iR*mR + iG*mG;
            mj = jR*mR + jG*mG;

            vrel = vel(i,:) - vel(j,:);
            vn = dot(vrel, n);

            if vn < 0
                e = e_GG;   % RG = GG
                j_imp = -(1+e) * vn / (1/mi + 1/mj);
                Jvec  = j_imp * n;

                vel(i,:) = vel(i,:) + (Jvec/mi);
                vel(j,:) = vel(j,:) - (Jvec/mj);

                J_RG = J_RG + norm(Jvec);  
            end

            % cooldown when you give cool dwon step
            % cooldown(i,j) = uint16(cooldown_steps);
            % cooldown(j,i) = uint16(cooldown_steps);
        end
    end
end

    % % ---------- (H) Enforce RR cluster COM velocity (fast DSU) ----------
    % if ~isempty(rr_edges)
    %     % Build DSU for red nodes only (but edges are RR so both red)
    %     parent = (1:N)';
    % 
    %     % union edges
    %     for ee = 1:size(rr_edges,1)
    %         a = rr_edges(ee,1);
    %         b = rr_edges(ee,2);
    %         parent = dsu_union(parent, a, b);
    %     end
    % 
    %     % compress and gather components for red only
    %     roots = zeros(N,1);
    %     for i = 1:N
    %         if isR(i)
    %             roots(i) = dsu_find(parent, i);
    %         end
    %     end
    % 
    %     % compute v_cm per root
    %     % Use accumarray on roots for x and y
    %     red_idx = find(isR & roots>0);
    %     rts = roots(red_idx);
    % 
    %     vx = vel(red_idx,1);
    %     vy = vel(red_idx,2);
    % 
    %     % map root ids to 1..K for accumarray (stable)
    %     [uRoots, ~, g] = unique(rts);
    %     sumVx = accumarray(g, vx);
    %     sumVy = accumarray(g, vy);
    %     cnt   = accumarray(g, 1);
    % 
    %     vcmx = sumVx ./ max(cnt,1);
    %     vcmy = sumVy ./ max(cnt,1);
    % 
    %     % assign back: all nodes in component share v_cm
    %     vel(red_idx,1) = vcmx(g);
    %     vel(red_idx,2) = vcmy(g);
    % 
    %     % collision intensity proxy for RR: how much change from original?
    %     % (cheap approximate: sum of component velocity variance)
    %     % We'll approximate by using within-component correction magnitude
    %     % Not exact momentum impulse, but OK as an “intensity”.
    %     J_RR = sum( hypot(vx - vcmx(g), vy - vcmy(g)) );
    % end
    % 
    % coll_RR(t) = J_RR;
    % coll_GG(t) = J_GG;
    % coll_RG(t) = J_RG;

    % ---------- (I) metrics ----------
    spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
    u   = vel ./ spd;

    Nr_now = sum(isR); Ng_now = sum(isG);
    pol_red(t)    = (Nr_now>0) * ( norm(sum(u(isR,:),1)) / max(Nr_now,1) );
    pol_green(t)  = (Ng_now>0) * ( norm(sum(u(isG,:),1)) / max(Ng_now,1) );
    pol_all(t)    = norm(sum(u,1)) / N;
    mean_speed(t) = mean(spd);

% ---------- (J) visualization ----------
if mod(t,plot_every)==0 || t==1

    cla(ax); hold(ax,'on'); axis(ax,'equal');

    % domain circle
    th = linspace(0,2*pi,400);
    plot(ax, R_domain*cos(th), R_domain*sin(th),'k-','LineWidth',1.5);

    % particles
    scatter(ax, pos(isR,1),pos(isR,2),22,[0.94 0.882 0.189],'filled');
    scatter(ax, pos(isG,1),pos(isG,2),22,'cyan','filled');

    % ---- Better vectors (clear & consistent length) ----
    spd_now = hypot(vel(:,1),vel(:,2));
    u_now   = vel ./ max(spd_now,1e-12);

    % show more arrows than before (e.g., 1:2:N)
    idx_show = 1:2:N;

    % fixed visual arrow length (in position units) so it “pops”
    arrow_len = 1.5;   % 0.8~1.8 
    vx = arrow_len * u_now(idx_show,1);
    vy = arrow_len * u_now(idx_show,2);

    q = quiver(ax, pos(idx_show,1), pos(idx_show,2), vx, vy, ...
               0, 'k', 'LineWidth', 1.0, 'MaxHeadSize', 3.2);
    % scale=0 :

    xlim(ax, [-R_domain-3, R_domain+3]);
    ylim(ax, [-R_domain-3, R_domain+3]);
    title(ax, sprintf('t=%d / %d', t, Tsteps));

    drawnow;

    if save_frames
    fname = fullfile(img_dir, sprintf('frame_%05d.%s', t, img_format));
    exportgraphics(gcf, fname, 'Resolution', dpi);
    end

    % if make_video
    %     frame = getframe(gcf);
    %     writeVideo(vwriter, frame);
    % end
end
end
disp("Done.");

%% ===================== HELPERS =====================

function pairs = grid_pairs_fast(pos, rcut, R_domain)
% Unique pairs (i<j) within distance < rcut using grid binning.

N = size(pos,1);
if N < 2
    pairs = zeros(0,2);
    return;
end

cellSize = rcut;
xmin = -R_domain; ymin = -R_domain;

nx = max(1, ceil((2*R_domain) / cellSize));
ny = nx;

ix = floor((pos(:,1) - xmin) / cellSize) + 1;
iy = floor((pos(:,2) - ymin) / cellSize) + 1;
ix = min(max(ix,1), nx);
iy = min(max(iy,1), ny);

cellId = ix + (iy-1)*nx;

[sortedCell, order] = sort(cellId);
posS   = pos(order,:);
orderS = order;

edges = [1; find(diff(sortedCell)~=0)+1; N+1];
cells = sortedCell(edges(1:end-1));

pairsCap = max(20000, 10*N);
pairs = zeros(pairsCap,2);
pcount = 0;

offsets = [-1 -1; 0 -1; 1 -1; -1 0; 0 0; 1 0; -1 1; 0 1; 1 1];

% map cellId -> index in cells (fast lookup)
cell_to_idx = containers.Map('KeyType','int32','ValueType','int32');
for k = 1:numel(cells)
    cell_to_idx(int32(cells(k))) = int32(k);
end

for cidx = 1:numel(cells)
    c = cells(cidx);

    a0 = edges(cidx);
    a1 = edges(cidx+1)-1;
    idxA = a0:a1;

    cx = mod(c-1, nx) + 1;
    cy = floor((c-1)/nx) + 1;

    for oo = 1:size(offsets,1)
        cx2 = cx + offsets(oo,1);
        cy2 = cy + offsets(oo,2);
        if cx2 < 1 || cx2 > nx || cy2 < 1 || cy2 > ny
            continue;
        end
        c2 = cx2 + (cy2-1)*nx;

        if c2 < c
            continue; % avoid duplicates
        end

        key = int32(c2);
        if ~isKey(cell_to_idx, key)
            continue;
        end
        jpos = double(cell_to_idx(key));

        b0 = edges(jpos);
        b1 = edges(jpos+1)-1;
        idxB = b0:b1;

        if c2 == c
            for ii = 1:numel(idxA)-1
                iS = idxA(ii);
                pi = posS(iS,:);
                jj = idxA(ii+1:end);
                d = hypot(posS(jj,1)-pi(1), posS(jj,2)-pi(2));
                hit = jj(d < rcut);
                if ~isempty(hit)
                    for hh = 1:numel(hit)
                        pcount = pcount + 1;
                        if pcount > size(pairs,1)
                            pairs = [pairs; zeros(pairsCap,2)]; %#ok<AGROW>
                        end
                        pairs(pcount,:) = [orderS(iS), orderS(hit(hh))];
                    end
                end
            end
        else
            for ii = 1:numel(idxA)
                iS = idxA(ii);
                pi = posS(iS,:);
                d = hypot(posS(idxB,1)-pi(1), posS(idxB,2)-pi(2));
                hit = idxB(d < rcut);
                if ~isempty(hit)
                    for hh = 1:numel(hit)
                        pcount = pcount + 1;
                        if pcount > size(pairs,1)
                            pairs = [pairs; zeros(pairsCap,2)]; %#ok<AGROW>
                        end
                        pairs(pcount,:) = [orderS(iS), orderS(hit(hh))];
                    end
                end
            end
        end
    end
end

pairs = pairs(1:pcount,:);
if isempty(pairs)
    pairs = zeros(0,2);
else
    pairs = sort(pairs,2);
    pairs = unique(pairs,'rows');
end
end

function y = normalize01(x)
x = x(:);
if all(isnan(x)) || numel(x)<2
    y = x; return;
end
xmin = min(x(~isnan(x)));
xmax = max(x(~isnan(x)));
if xmax - xmin < 1e-12
    y = 0*x;
else
    y = (x - xmin) / (xmax - xmin);
end
end

% -------- DSU (Union-Find) --------
function parent = dsu_union(parent, a, b)
ra = dsu_find(parent, a);
rb = dsu_find(parent, b);
if ra ~= rb
    parent(rb) = ra;
end
end

function r = dsu_find(parent, a)
r = a;
while parent(r) ~= r
    r = parent(r);
end
% path compression
while parent(a) ~= a
    p = parent(a);
    parent(a) = r;
    a = p;
end
end
