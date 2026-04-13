%% run_red_sweep_phase_diagram.m
% Red fraction sweep: p_red = 0:0.1:1.0, each 10 reps
% Save:
%  (1) phase_timeseries_redsweep_0323_50.csv : sampled time series
%  (2) phase_lastframe_redsweep_0323_50.csv  : last-frame summary
%  (3) phase_localhet_redsweep_0323_50.csv   : local heterogeneity summary
%
% Model:
% - Circular confinement with soft wall
% - OU persistent acceleration noise
% - Type-specific Vicsek-B
% - Collisions:
%   RR: one-shot inelastic event (COM velocity)
%   GG: elastic
%   RG: optional elastic like GG

clear; clc;

%% ===================== SWEEP SETTINGS =====================
p_list   = 0:0.1:1.0;
n_reps   = 10;
baseSeed = 130;

out_csv_ts   = "phase_timeseries_redsweep_0326_10frame_50.csv";
out_csv_last = "phase_lastframe_redsweep_0326_10frame_50.csv";
out_csv_het  = "phase_localhet_redsweep_0326_10frame_50.csv";

%% ===================== SHARED PARAMETERS =====================
P = struct();

% Domain
P.R_domain = 30;
P.k_wall   = 50.0;

% Simulation
P.N      = 600;
P.dt     = 0.05;
P.Tsteps = 2000;

% Sampling for outputs
P.sample_every = 50;

% NEW: window length for metrics
P.metric_window = 10;   % use 10-step net displacement

% Local heterogeneity settings
P.compute_localhet = true;
P.r_het = 3.5;
P.localhet_steps = [];

% Speeds
P.v_init = 1.0;
P.vmax   = 2.5;

% Drag (optional)
P.gamma_R = 0.0;
P.gamma_G = 0.0;

% OU persistent noise
P.use_OU_noise = true;
P.tau_R   = 1.0;
P.tau_G   = 1.0;
P.sigma_R = 0.7;
P.sigma_G = 0.7;

% Optional RR alignment accel
P.r_align = 0.0;
P.J_align = 0.0;

% Collisions
P.r_coll = 2.0;
P.pos_correction = 0.8;
P.eps_sep = 0.001;

P.cooldown_steps = 0;
P.RR_require_approach = true;

P.mR = 1.0;
P.mG = 1.0;

P.enable_GG_collision = true;
P.e_GG = 1.0;

% RG rule
P.RG_same_as_GG = true;
P.cooldown_only_RG = true;

% Vicsek-B
P.use_vicsek_B = true;

% Red Vicsek
P.r_vicsek_R   = 2.0;
P.alpha_v_R    = 0.1;
P.p_vicsek_R   = 1.0;
P.vicsek_warmup_R = 0;

% Green Vicsek
P.r_vicsek_G   = 2.0;
P.alpha_v_G    = 0.1;
P.p_vicsek_G   = 1.0;
P.vicsek_warmup_G = 0;

%% Correlation length settings
C = struct();
C.maxPairs   = 50000;
C.nBins      = 40;
C.rMax       = P.R_domain;
C.useUnitVel = true;   % now means: use unit displacement direction

%% ===================== RUN SWEEP =====================
TS_all   = table();
LAST_all = table();
HET_all  = table();

row_last = 0;

for ip = 1:numel(p_list)
    p_red = p_list(ip);

    for rep = 1:n_reps
        seed = baseSeed + 1000*ip + rep;
        P.p_red = p_red;

        out = simulate_one_run(P, C, seed);

        % ---- append timeseries ----
        ts = out.ts;
        ts.p_red = repmat(p_red, height(ts), 1);
        ts.rep   = repmat(rep,  height(ts), 1);
        ts.seed  = repmat(seed, height(ts), 1);
        TS_all = [TS_all; ts]; %#ok<AGROW>

        % ---- append last-frame ----
        row_last = row_last + 1;
        LAST_all.p_red(row_last,1)           = p_red;
        LAST_all.rep(row_last,1)             = rep;
        LAST_all.seed(row_last,1)            = seed;
        LAST_all.mean_speed_last(row_last,1) = out.mean_speed_last;
        LAST_all.corr_len_last(row_last,1)   = out.corr_len_last;
        LAST_all.pol_all_last(row_last,1)    = out.pol_all_last;
        LAST_all.Nr(row_last,1)              = out.Nr;
        LAST_all.Ng(row_last,1)              = out.Ng;

        % ---- append local heterogeneity ----
        if isfield(out, "localhet") && ~isempty(out.localhet)
            ht = out.localhet;
            ht.p_red = repmat(p_red, height(ht), 1);
            ht.rep   = repmat(rep,  height(ht), 1);
            ht.seed  = repmat(seed, height(ht), 1);
            ht.Nr    = repmat(out.Nr, height(ht), 1);
            ht.Ng    = repmat(out.Ng, height(ht), 1);
            HET_all = [HET_all; ht]; %#ok<AGROW>
        end

        fprintf("[DONE] p_red=%.1f rep=%d seed=%d | v=%.3f xi=%.3f pol=%.3f\n", ...
            p_red, rep, seed, out.mean_speed_last, out.corr_len_last, out.pol_all_last);
    end
end

writetable(TS_all, out_csv_ts);
writetable(LAST_all, out_csv_last);
writetable(HET_all, out_csv_het);

disp("Saved: " + out_csv_ts);
disp("Saved: " + out_csv_last);
disp("Saved: " + out_csv_het);

G = groupsummary(LAST_all, "p_red", ["mean","std"], ...
    ["mean_speed_last","corr_len_last","pol_all_last"]);
writetable(G, "phase_lastframe_redsweep_summary.csv");
disp("Saved: phase_lastframe_redsweep_summary.csv");


%% ===================== FUNCTIONS =====================

function out = simulate_one_run(P, C, seed)
rng(seed);

N  = P.N;
dt = P.dt;

Nr = round(N * P.p_red);
Ng = N - Nr;

types = [ones(Nr,1); zeros(Ng,1)];   % 1=Red, 0=Green
isR = (types==1);
isG = ~isR;

% init positions uniform in disk
theta = 2*pi*rand(N,1);
rad   = P.R_domain * sqrt(rand(N,1));
pos   = [rad.*cos(theta), rad.*sin(theta)];

% init velocities random direction
dir0 = 2*pi*rand(N,1);
vel  = P.v_init * [cos(dir0), sin(dir0)];

% OU state
ou = zeros(N,2);

% cooldown matrix
cooldown = zeros(N,N,'uint16');

% time series sampling buffers
sample_every = P.sample_every;
nsamp = floor(P.Tsteps / sample_every);
ts_step  = zeros(nsamp,1);
ts_time  = zeros(nsamp,1);
ts_speed = nan(nsamp,1);
ts_xi    = nan(nsamp,1);
ts_pol   = nan(nsamp,1);
sidx = 0;

% instantaneous metrics accumulators (kept for internal reference only)
pol_all_inst    = nan(P.Tsteps,1);
mean_speed_inst = nan(P.Tsteps,1);

% metric window
metric_window = P.metric_window;
pos_hist = cell(metric_window + 1, 1);

% ---------- local heterogeneity schedule ----------
doHet = isfield(P,"compute_localhet") && P.compute_localhet;
het_steps = [];
if doHet
    if isfield(P,"localhet_steps") && ~isempty(P.localhet_steps)
        het_steps = unique(P.localhet_steps(:)');
    else
        het_steps = unique([1, round(P.Tsteps/2), P.Tsteps]);
    end
end
het_done = false(size(het_steps));
het_rows = 0;
localhet_tbl = table();
% -----------------------------------------------

for t = 1:P.Tsteps

    % store current position BEFORE update into ring buffer
    buf_idx = mod(t-1, metric_window + 1) + 1;
    pos_hist{buf_idx} = pos;

    % cooldown decrement
    if any(cooldown(:) > 0)
        cooldown(cooldown > 0) = cooldown(cooldown > 0) - 1;
    end

    % ---------- (A) RR alignment accel ----------
    spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
    u   = vel ./ spd;
    align_acc = zeros(N,2);

    if P.J_align > 0 && Nr > 1
        pairsA = grid_pairs_fast(pos, P.r_align, P.R_domain);
        if ~isempty(pairsA)
            sumUx = zeros(N,1); sumUy = zeros(N,1); cnt = zeros(N,1);
            i = pairsA(:,1); j = pairsA(:,2);

            RRmask = isR(i) & isR(j);
            iRR = i(RRmask); jRR = j(RRmask);

            sumUx(iRR) = sumUx(iRR) + u(jRR,1); sumUy(iRR) = sumUy(iRR) + u(jRR,2); cnt(iRR) = cnt(iRR) + 1;
            sumUx(jRR) = sumUx(jRR) + u(iRR,1); sumUy(jRR) = sumUy(jRR) + u(iRR,2); cnt(jRR) = cnt(jRR) + 1;

            idx = isR & (cnt > 0);
            meanDir = [sumUx(idx)./cnt(idx), sumUy(idx)./cnt(idx)];
            align_acc(idx,:) = P.J_align * meanDir;
        end
    end

    % ---------- (B) wall ----------
    rnorm = hypot(pos(:,1),pos(:,2));
    over  = max(0, rnorm - P.R_domain);
    wall_dir = -[pos(:,1)./max(rnorm,1e-9), pos(:,2)./max(rnorm,1e-9)];
    wall_acc = (P.k_wall * over) .* wall_dir;

    % ---------- (C) OU noise ----------
    if P.use_OU_noise
        tau = P.tau_R*ones(N,1);   tau(isG) = P.tau_G;
        sig = P.sigma_R*ones(N,1); sig(isG) = P.sigma_G;

        a  = exp(-dt ./ max(tau,1e-9));
        ou = ou .* a + (sig .* sqrt(1 - a.^2)) .* randn(N,2);

        noise_acc = ou;
    else
        noise_acc = zeros(N,2);
    end

    % ---------- (D) drag ----------
    gamma = P.gamma_R*ones(N,1); gamma(isG) = P.gamma_G;
    drag_acc = -gamma .* vel;

    % ---------- (E) integrate ----------
    acc = align_acc + wall_acc + noise_acc + drag_acc;

    max_acc = 12.0;
    an = max(hypot(acc(:,1),acc(:,2)), 1e-12);
    acc = acc .* min(1, max_acc ./ an);

    vel = vel + dt * acc;

    % speed cap
    spd = hypot(vel(:,1),vel(:,2));
    tooFast = spd > P.vmax;
    if any(tooFast)
        vel(tooFast,:) = vel(tooFast,:) .* (P.vmax ./ spd(tooFast));
    end

    pos = pos + dt * vel;

    % outside correction
    rnorm = hypot(pos(:,1),pos(:,2));
    outside = rnorm > (P.R_domain + 0.5);
    if any(outside)
        corr_dir = -[pos(outside,1)./rnorm(outside), pos(outside,2)./rnorm(outside)];
        pos(outside,:) = pos(outside,:) + 0.2 * corr_dir;
    end

    % ---------- (F) Vicsek-B ----------
    if P.use_vicsek_B
        if t > P.vicsek_warmup_R && P.alpha_v_R > 0 && any(isR)
            vel = apply_vicsek(pos, vel, isR, P.r_vicsek_R, P.alpha_v_R, P.p_vicsek_R, P.R_domain);
        end
        if t > P.vicsek_warmup_G && P.alpha_v_G > 0 && any(isG)
            vel = apply_vicsek(pos, vel, isG, P.r_vicsek_G, P.alpha_v_G, P.p_vicsek_G, P.R_domain);
        end
    end

    % ---------- (G) collisions ----------
    pairsC = grid_pairs_fast(pos, P.r_coll, P.R_domain);

    if ~isempty(pairsC)
        for kk = 1:size(pairsC,1)
            i = pairsC(kk,1);
            j = pairsC(kk,2);

            dx = pos(i,1) - pos(j,1);
            dy = pos(i,2) - pos(j,2);
            d  = hypot(dx,dy);

            if d <= 1e-12 || d >= P.r_coll
                continue;
            end

            n = [dx, dy] / d;
            overlap = P.r_coll - d;

            % soft overlap correction
            if overlap > 0
                corr = 0.5 * P.pos_correction * (overlap + P.eps_sep) * n;
                pos(i,:) = pos(i,:) + corr;
                pos(j,:) = pos(j,:) - corr;
            end

            iR = isR(i); jR = isR(j);
            iG = ~iR;    jG = ~jR;
            isRG = (iR && jG) || (iG && jR);

            if ((iR && jR) || (P.cooldown_only_RG && isRG)) && cooldown(i,j) > 0
                continue;
            end

            if iR && jR
                vrel = vel(i,:) - vel(j,:);
                vn = dot(vrel, n);
                if P.RR_require_approach && vn >= 0
                    continue;
                end

                vcm = (P.mR*vel(i,:) + P.mR*vel(j,:)) / (2*P.mR);
                vel(i,:) = vcm;
                vel(j,:) = vcm;

                cooldown(i,j) = uint16(P.cooldown_steps);
                cooldown(j,i) = uint16(P.cooldown_steps);

            elseif iG && jG
                if ~P.enable_GG_collision
                    continue;
                end

                vrel = vel(i,:) - vel(j,:);
                vn = dot(vrel, n);
                if vn < 0
                    mi = P.mG; mj = P.mG;
                    j_imp = -(1 + P.e_GG) * vn / (1/mi + 1/mj);
                    Jvec  = j_imp * n;
                    vel(i,:) = vel(i,:) + (Jvec/mi);
                    vel(j,:) = vel(j,:) - (Jvec/mj);
                end

            else
                if P.RG_same_as_GG
                    vrel = vel(i,:) - vel(j,:);
                    vn = dot(vrel, n);
                    if vn < 0
                        mi = iR*P.mR + iG*P.mG;
                        mj = jR*P.mR + jG*P.mG;
                        e  = P.e_GG;
                        j_imp = -(1 + e) * vn / (1/mi + 1/mj);
                        Jvec  = j_imp * n;
                        vel(i,:) = vel(i,:) + (Jvec/mi);
                        vel(j,:) = vel(j,:) - (Jvec/mj);
                    end
                end

                if P.cooldown_only_RG
                    cooldown(i,j) = uint16(P.cooldown_steps);
                    cooldown(j,i) = uint16(P.cooldown_steps);
                end
            end
        end
    end

    % ---------- instantaneous metrics (not exported) ----------
    spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
    u   = vel ./ spd;
    pol_all_inst(t)    = norm(sum(u,1)) / N;
    mean_speed_inst(t) = mean(spd);

    % ---------- windowed metrics ----------
    if mod(t, sample_every) == 0
        sidx = sidx + 1;
        ts_step(sidx) = t;
        ts_time(sidx) = t * dt;

        if t > metric_window
            old_idx = mod(t - metric_window - 1, metric_window + 1) + 1;
            pos_old = pos_hist{old_idx};

            dpos_win = pos - pos_old;
            disp_mag = hypot(dpos_win(:,1), dpos_win(:,2));

            speed_win = disp_mag / (metric_window * dt);
            u_win = dpos_win ./ max(disp_mag, 1e-12);

            ts_speed(sidx) = mean(speed_win);
            ts_pol(sidx)   = norm(sum(u_win,1)) / N;
            ts_xi(sidx)    = correlation_length_displacement(pos, dpos_win, C);
        else
            ts_speed(sidx) = NaN;
            ts_pol(sidx)   = NaN;
            ts_xi(sidx)    = NaN;
        end
    end

    % ---------- local heterogeneity ----------
    if doHet && ~isempty(het_steps)
        k = find(~het_done & (t == het_steps), 1, 'first');
        if ~isempty(k)
            het_done(k) = true;

            r_het = P.r_het;
            het = local_heterogeneity_bruteforce(pos, types, r_het);

            het_rows = het_rows + 1;
            localhet_tbl.step(het_rows,1)  = t;
            localhet_tbl.time(het_rows,1)  = t * dt;
            localhet_tbl.mean_localhet_all(het_rows,1)   = het.mean_all;
            localhet_tbl.std_localhet_all(het_rows,1)    = het.std_all;
            localhet_tbl.median_localhet_all(het_rows,1) = het.median_all;
            localhet_tbl.mean_localhet_Rref(het_rows,1)  = het.mean_Rref;
            localhet_tbl.mean_localhet_Gref(het_rows,1)  = het.mean_Gref;
            localhet_tbl.frac_no_neighbors(het_rows,1)   = het.frac_no_neighbors;
            localhet_tbl.mean_n_neighbors(het_rows,1)    = het.mean_n_neighbors;
            localhet_tbl.r_het(het_rows,1)               = r_het;
        end
    end
end

% ---------- last-frame summary with windowed metrics ----------
if P.Tsteps > metric_window
    old_idx = mod(P.Tsteps - metric_window - 1, metric_window + 1) + 1;
    pos_old = pos_hist{old_idx};

    dpos_last = pos - pos_old;
    disp_mag_last = hypot(dpos_last(:,1), dpos_last(:,2));
    speed_last = disp_mag_last / (metric_window * dt);
    u_last = dpos_last ./ max(disp_mag_last, 1e-12);

    out.mean_speed_last = mean(speed_last);
    out.pol_all_last    = norm(sum(u_last,1)) / N;
    out.corr_len_last   = correlation_length_displacement(pos, dpos_last, C);
else
    out.mean_speed_last = NaN;
    out.pol_all_last    = NaN;
    out.corr_len_last   = NaN;
end

out.Nr = Nr;
out.Ng = Ng;

out.ts = table(ts_step(1:sidx), ts_time(1:sidx), ts_speed(1:sidx), ts_xi(1:sidx), ts_pol(1:sidx), ...
    'VariableNames', {'step','time','mean_speed','corr_len','pol_all'});

if doHet
    out.localhet = localhet_tbl;
else
    out.localhet = table();
end

end


function vel = apply_vicsek(pos, vel, targetMask, r_v, alpha_v, p_apply, R_domain)
N = size(pos,1);
pairsV = grid_pairs_fast(pos, r_v, R_domain);
if isempty(pairsV), return; end

spd = max(hypot(vel(:,1),vel(:,2)), 1e-12);
u   = vel ./ spd;

sumUx = zeros(N,1); sumUy = zeros(N,1); cnt = zeros(N,1);
i = pairsV(:,1); j = pairsV(:,2);

sumUx(i) = sumUx(i) + u(j,1); sumUy(i) = sumUy(i) + u(j,2); cnt(i) = cnt(i) + 1;
sumUx(j) = sumUx(j) + u(i,1); sumUy(j) = sumUy(j) + u(i,2); cnt(j) = cnt(j) + 1;

idx = targetMask & (cnt > 0);
if p_apply < 1
    idx = idx & (rand(N,1) < p_apply);
end
if ~any(idx), return; end

md = [sumUx(idx)./cnt(idx), sumUy(idx)./cnt(idx)];
md = md ./ max(hypot(md(:,1),md(:,2)), 1e-12);

vmag = spd(idx);
v_align = md .* vmag;

vel(idx,:) = (1-alpha_v)*vel(idx,:) + alpha_v*v_align;
end


function xi = correlation_length_displacement(pos, dpos, C)
% C(r) = < u_i dot u_j > using windowed displacement direction
% xi = first r where C(r) <= C0/e

N = size(pos,1);

if C.useUnitVel
    mag = max(hypot(dpos(:,1), dpos(:,2)), 1e-12);
    u = dpos ./ mag;
else
    u = dpos;
end

M = C.maxPairs;
idx1 = randi(N, M, 1);
idx2 = randi(N, M, 1);
same = idx1 == idx2;
idx2(same) = mod(idx2(same), N) + 1;

dp = sum(u(idx1,:).*u(idx2,:), 2);
dr = pos(idx1,:) - pos(idx2,:);
r  = hypot(dr(:,1), dr(:,2));

edges = linspace(0, C.rMax, C.nBins+1);
[~,~,bin] = histcounts(r, edges);

Cbin = nan(C.nBins,1);
for b = 1:C.nBins
    m = (bin == b);
    if any(m)
        Cbin(b) = mean(dp(m));
    end
end

b0 = find(~isnan(Cbin), 1, 'first');
if isempty(b0)
    xi = NaN; return;
end

C0 = Cbin(b0);
if abs(C0) < 1e-12
    xi = NaN; return;
end

thr = C0 / exp(1);

xi = NaN;
for b = b0+1:C.nBins
    if ~isnan(Cbin(b)) && Cbin(b) <= thr
        xi = 0.5 * (edges(b) + edges(b+1));
        return;
    end
end
end


function pairs = grid_pairs_fast(pos, rcut, R_domain)
N = size(pos,1);
if N < 2, pairs = zeros(0,2); return; end

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
            continue;
        end

        key = int32(c2);
        if ~isKey(cell_to_idx, key), continue; end
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
                for hh = 1:numel(hit)
                    pcount = pcount + 1;
                    if pcount > size(pairs,1)
                        pairs = [pairs; zeros(pairsCap,2)]; %#ok<AGROW>
                    end
                    pairs(pcount,:) = [orderS(iS), orderS(hit(hh))];
                end
            end
        else
            for ii = 1:numel(idxA)
                iS = idxA(ii);
                pi = posS(iS,:);
                d = hypot(posS(idxB,1)-pi(1), posS(idxB,2)-pi(2));
                hit = idxB(d < rcut);
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

pairs = pairs(1:pcount,:);
if isempty(pairs)
    pairs = zeros(0,2);
else
    pairs = sort(pairs,2);
    pairs = unique(pairs,'rows');
end
end


function het = local_heterogeneity_bruteforce(pos, types, r_het)

N = size(pos,1);

nNbr  = zeros(N,1);
nDiff = zeros(N,1);

for i = 1:N
    dx = pos(:,1) - pos(i,1);
    dy = pos(:,2) - pos(i,2);
    d  = hypot(dx, dy);

    neighbors = (d < r_het);
    neighbors(i) = false;

    nNbr(i) = sum(neighbors);

    if nNbr(i) > 0
        nDiff(i) = sum(types(neighbors) ~= types(i));
    end
end

localHet = nan(N,1);
hasNbr = (nNbr > 0);
localHet(hasNbr) = nDiff(hasNbr) ./ nNbr(hasNbr);

vals = localHet(hasNbr);

if isempty(vals)
    het.mean_all = NaN;
    het.std_all  = NaN;
    het.median_all = NaN;
    het.frac_no_neighbors = 1.0;
    het.mean_n_neighbors  = 0.0;
    het.mean_Rref = NaN;
    het.mean_Gref = NaN;
    return;
end

het.mean_all   = mean(vals);
het.std_all    = std(vals);
het.median_all = median(vals);

isR = (types==1);
isG = ~isR;

het.mean_Rref = mean(localHet(hasNbr & isR), 'omitnan');
het.mean_Gref = mean(localHet(hasNbr & isG), 'omitnan');

het.frac_no_neighbors = mean(~hasNbr);
het.mean_n_neighbors  = mean(nNbr);

end