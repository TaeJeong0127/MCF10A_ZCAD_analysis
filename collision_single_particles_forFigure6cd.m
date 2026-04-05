%% simulation_collisionsimple_dispbetween_OUturn_WITHMP4_preNorm.m
% Displacement-based between-angle + reorientation per-particle
% + pre-collision between angle
% + normalization by theta_pre_between
% + MP4 visualization with collision markers

clear; clc;

%% ===================== USER SETTINGS =====================
S = struct();

% population
S.N      = 140;
S.p_red  = 0.5;
S.steps  = 5000;
S.dt     = 0.2;

% domain (box)
S.box = [0 120 0 120];
S.r   = 1.0;

% motion (active)
S.v0   = 1.3;
S.vmax = 3.0;
S.friction = 0.0;

% OU turning
S.use_OU_turn = true;
S.tau_turn    = 6.0;
S.sigma_turn  = 0.20;

% integration
S.substeps        = 1;
S.max_coll_iters  = 2;

% collisions
S.GG_elastic = true;
S.RG_elastic = true;

% RR stick+release
S.RR_stick_steps    = 6;
S.RR_release_mode   = "OUkick";     % "random" or "OUkick"
S.RR_release_kick_sigma = 0.9;
S.RR_release_speed  = "keep";       % "keep" or "reset"
S.RR_cooldown_steps = 8;

% angles (logging)
S.delay_steps   = 10;
S.min_post_disp = 0.2;
S.min_norm      = 1e-12;
S.grid_cell     = 3.0;

% normalization safety
S.min_pre_between_deg = 1e-6;

% outputs (angles)
S.out_csv = "collision_angles_dispbetween_OU_with_pre_norm.csv";
S.out_fig = "collision_angles_dispbetween_OU_with_pre_norm.png";

% ===== MP4 SETTINGS =====
S.make_mp4   = true;
S.mp4_path   = "simulation_OU_collisions.mp4";
S.mp4_every  = 2;        % save frame every N simulation steps
S.mp4_fps    = 30;
S.mp4_quality= 90;

% ===== Collision marker (for movie) =====
S.show_coll_mark = true;
S.coll_mark_ttl  = 10;    % TTL in "saved frames" units
S.coll_mark_size = 80;
S.max_marks      = 2000;

%% ===================== INIT =====================
N = S.N;
xmin=S.box(1); xmax=S.box(2); ymin=S.box(3); ymax=S.box(4);

Nr = round(N*S.p_red);
types = [ones(Nr,1); zeros(N-Nr,1)];
types = types(randperm(N));

% non-overlap init
pos = zeros(N,2);
for ii=1:N
    ok=false; tries=0;
    while ~ok
        tries = tries + 1;
        if tries > 12000
            error("Initial placement failed. Reduce N or r, enlarge box.");
        end
        pos(ii,:) = [xmin + (xmax-xmin)*rand, ymin + (ymax-ymin)*rand];
        if ii==1
            ok=true;
        else
            d = hypot(pos(1:ii-1,1)-pos(ii,1), pos(1:ii-1,2)-pos(ii,2));
            ok = all(d > 2.2*S.r);
        end
    end
end

theta = 2*pi*rand(N,1);
speed = S.v0 * ones(N,1);
omega = zeros(N,1);

pos_prev = pos;

stick_left    = zeros(N,1,'int32');
stick_partner = zeros(N,1,'int32');
rr_cooldown   = zeros(N,N,'uint16');

% collision mark buffer
mark_pos = zeros(0,2);
mark_ttl = zeros(0,1);

%% ===================== EVENT QUEUE (for delayed eval) =====================
delay = S.delay_steps;
capEv = 400000;
ev_i        = zeros(capEv,1,'int32');
ev_j        = zeros(capEv,1,'int32');
ev_step0    = zeros(capEv,1,'int32');
ev_pairType = strings(capEv,1);

ev_pos_i0      = zeros(capEv,2);
ev_pos_j0      = zeros(capEv,2);
ev_pos_i_prev  = zeros(capEv,2);
ev_pos_j_prev  = zeros(capEv,2);

% NEW: pre-collision displacement vectors
ev_pre_di      = zeros(capEv,2);
ev_pre_dj      = zeros(capEv,2);

% NEW: true pre-collision speed (instantaneous, before response)
ev_speed_i_pre = zeros(capEv,1);
ev_speed_j_pre = zeros(capEv,1);
ev_vx_i_pre    = zeros(capEv,1);
ev_vy_i_pre    = zeros(capEv,1);
ev_vx_j_pre    = zeros(capEv,1);
ev_vy_j_pre    = zeros(capEv,1);

ev_count    = 0;

%% ===================== OUTPUT BUFFERS =====================
capOut = 800000;
rows = 0;

out_pairType = strings(capOut,1);
out_step0    = zeros(capOut,1);
out_stepEval = zeros(capOut,1);
out_i        = zeros(capOut,1,'int32');
out_j        = zeros(capOut,1,'int32');
out_ref      = strings(capOut,1);

out_theta_pre_between   = zeros(capOut,1);
out_theta_between       = zeros(capOut,1);
out_theta_reorient      = zeros(capOut,1);
out_theta_between_norm  = zeros(capOut,1);
out_theta_reorient_norm = zeros(capOut,1);

% NEW: pre-collision speed outputs
out_speed_pre_i   = zeros(capOut,1);
out_speed_pre_j   = zeros(capOut,1);
out_speed_pre_ref = zeros(capOut,1);
out_speed_pre_opp = zeros(capOut,1);

out_vx_pre_i   = zeros(capOut,1);
out_vy_pre_i   = zeros(capOut,1);
out_vx_pre_j   = zeros(capOut,1);
out_vy_pre_j   = zeros(capOut,1);
out_vx_pre_ref = zeros(capOut,1);
out_vy_pre_ref = zeros(capOut,1);
out_vx_pre_opp = zeros(capOut,1);
out_vy_pre_opp = zeros(capOut,1);

%% ===================== MP4 INIT =====================
if S.make_mp4
    vobj = VideoWriter(S.mp4_path, 'MPEG-4');
    vobj.FrameRate = S.mp4_fps;
    vobj.Quality   = S.mp4_quality;
    open(vobj);
    figure('Color','w');
end

%% ===================== MAIN LOOP =====================
for step = 1:S.steps

    if any(rr_cooldown(:) > 0)
        rr_cooldown(rr_cooldown>0) = rr_cooldown(rr_cooldown>0) - 1;
    end

    [theta, speed, omega, stick_left, stick_partner, rr_cooldown] = enforce_rr_stick_and_release_OU( ...
        theta, speed, omega, stick_left, stick_partner, rr_cooldown, S);

    for ss=1:S.substeps
        dt = S.dt / S.substeps;

        if ss==1
            pos_prev = pos;
        end

        % OU turning
        if S.use_OU_turn
            a = exp(-dt / max(S.tau_turn, 1e-9));
            omega = omega .* a + (S.sigma_turn * sqrt(1 - a^2)) .* randn(N,1);
            theta = theta + omega * dt;
        end

        speed = min(speed, S.vmax);
        vel = [speed.*cos(theta), speed.*sin(theta)];
        pos = pos + dt * vel;

        [pos, theta] = reflect_walls_theta(pos, theta, S);

        for it=1:S.max_coll_iters
            pairs = grid_pairs_box(pos, S.grid_cell, S.r, S.box);
            if isempty(pairs), break; end

            for kk=1:size(pairs,1)
                i = double(pairs(kk,1));
                j = double(pairs(kk,2));

                % ---- SAFETY: invalid indices guard ----
                if i < 1 || j < 1 || i > N || j > N || ~isfinite(i) || ~isfinite(j)
                    continue;
                end

                if stick_left(i)>0 || stick_left(j)>0
                    continue;
                end

                dx = pos(i,1) - pos(j,1);
                dy = pos(i,2) - pos(j,2);
                d  = hypot(dx,dy);

                if d <= S.min_norm || d >= 2*S.r
                    continue;
                end

                n = [dx, dy]/d;
                overlap = 2*S.r - d;

                % position correction
                corr = 0.5 * overlap * n;
                pos(i,:) = pos(i,:) + corr;
                pos(j,:) = pos(j,:) - corr;

                iR = (types(i)==1);
                jR = (types(j)==1);
                if iR && jR
                    pairType = "RR";
                elseif (~iR) && (~jR)
                    pairType = "GG";
                else
                    pairType = "RG";
                end

                vi_pre = [speed(i)*cos(theta(i)), speed(i)*sin(theta(i))];
                vj_pre = [speed(j)*cos(theta(j)), speed(j)*sin(theta(j))];
                vrel = vi_pre - vj_pre;
                vn = dot(vrel, n);
                if vn >= 0
                    continue; % not approaching
                end

                % ---- collision response ----
                if pairType == "RR"
                    if rr_cooldown(i,j) > 0
                        continue;
                    end

                    vcm = 0.5*(vi_pre + vj_pre);
                    sp_cm = max(norm(vcm), S.min_norm);
                    th_cm = atan2(vcm(2), vcm(1));
                    theta(i)=th_cm; theta(j)=th_cm;
                    speed(i)=sp_cm; speed(j)=sp_cm;

                    stick_left(i)=int32(S.RR_stick_steps);
                    stick_left(j)=int32(S.RR_stick_steps);
                    stick_partner(i)=int32(j);
                    stick_partner(j)=int32(i);

                elseif pairType == "GG"
                    if ~S.GG_elastic, continue; end
                    [vi_new, vj_new] = elastic_impulse_equalmass(vi_pre, vj_pre, n);
                    [theta(i), speed(i)] = vec_to_theta_speed(vi_new, S.min_norm);
                    [theta(j), speed(j)] = vec_to_theta_speed(vj_new, S.min_norm);

                else % RG
                    if ~S.RG_elastic, continue; end
                    [vi_new, vj_new] = elastic_impulse_equalmass(vi_pre, vj_pre, n);
                    [theta(i), speed(i)] = vec_to_theta_speed(vi_new, S.min_norm);
                    [theta(j), speed(j)] = vec_to_theta_speed(vj_new, S.min_norm);
                end

                speed(i) = min(speed(i), S.vmax);
                speed(j) = min(speed(j), S.vmax);

                % ---- add collision marker ----
                if S.make_mp4 && S.show_coll_mark
                    mark_pos(end+1,:) = 0.5*(pos(i,:)+pos(j,:)); %#ok<SAGROW>
                    mark_ttl(end+1,1) = S.coll_mark_ttl;         %#ok<SAGROW>
                    if numel(mark_ttl) > S.max_marks
                        mark_pos = mark_pos(end-S.max_marks+1:end,:);
                        mark_ttl = mark_ttl(end-S.max_marks+1:end,:);
                    end
                end

                % ---- queue event for delayed evaluation ----
                ev_count = ev_count + 1;
                if ev_count <= capEv
                    ev_i(ev_count)        = int32(i);
                    ev_j(ev_count)        = int32(j);
                    ev_step0(ev_count)    = int32(step);
                    ev_pairType(ev_count) = pairType;

                    ev_pos_i0(ev_count,:)     = pos(i,:);
                    ev_pos_j0(ev_count,:)     = pos(j,:);
                    ev_pos_i_prev(ev_count,:) = pos_prev(i,:);
                    ev_pos_j_prev(ev_count,:) = pos_prev(j,:);

                    % pre-collision displacement vectors
                    ev_pre_di(ev_count,:) = pos(i,:) - pos_prev(i,:);
                    ev_pre_dj(ev_count,:) = pos(j,:) - pos_prev(j,:);

                    % NEW: true pre-collision instantaneous velocity
                    ev_speed_i_pre(ev_count) = norm(vi_pre);
                    ev_speed_j_pre(ev_count) = norm(vj_pre);

                    ev_vx_i_pre(ev_count) = vi_pre(1);
                    ev_vy_i_pre(ev_count) = vi_pre(2);
                    ev_vx_j_pre(ev_count) = vj_pre(1);
                    ev_vy_j_pre(ev_count) = vj_pre(2);
                end
            end
        end
    end

    % ---- evaluate due events ----
    if ev_count > 0
        due = find(double(ev_step0(1:ev_count)) + delay == step);
        if ~isempty(due)
            for qq=1:numel(due)
                ee = due(qq);

                i = double(ev_i(ee)); 
                j = double(ev_j(ee));
                if i<1 || j<1 || i>N || j>N
                    continue;
                end

                di1 = pos(i,:) - ev_pos_i0(ee,:);
                dj1 = pos(j,:) - ev_pos_j0(ee,:);

                ni = norm(di1); 
                nj = norm(dj1);
                if ni < S.min_post_disp || nj < S.min_post_disp
                    continue;
                end

                % post-collision between angle
                cB = max(-1, min(1, dot(di1,dj1)/(ni*nj)));
                theta_between = acos(cB) * 180/pi;

                % pre-collision displacement vectors
                di0 = ev_pre_di(ee,:);
                dj0 = ev_pre_dj(ee,:);

                ni0 = norm(di0);
                nj0 = norm(dj0);

                % pre-collision between angle
                if ni0 < S.min_norm || nj0 < S.min_norm
                    theta_pre_between = NaN;
                else
                    cPre = max(-1, min(1, dot(di0,dj0)/(ni0*nj0)));
                    theta_pre_between = acos(cPre) * 180/pi;
                end

                % reorientation angles
                th_re_i = angle_between_vec(di0, di1, S.min_norm);
                th_re_j = angle_between_vec(dj0, dj1, S.min_norm);

                % normalization by theta_pre_between
                if isnan(theta_pre_between) || abs(theta_pre_between) < S.min_pre_between_deg
                    theta_between_norm = NaN;
                    th_re_i_norm = NaN;
                    th_re_j_norm = NaN;
                else
                    theta_between_norm = theta_between / theta_pre_between;
                    th_re_i_norm = th_re_i / theta_pre_between;
                    th_re_j_norm = th_re_j / theta_pre_between;
                end

                rows = rows + 1; 
                if rows>capOut, rows=capOut; break; end
                out_pairType(rows)=ev_pairType(ee);
                out_step0(rows)=double(ev_step0(ee));
                out_stepEval(rows)=step;
                out_i(rows)=int32(i); 
                out_j(rows)=int32(j);
                out_ref(rows)="i";
                out_theta_pre_between(rows)=theta_pre_between;
                out_theta_between(rows)=theta_between;
                out_theta_reorient(rows)=th_re_i;
                out_theta_between_norm(rows)=theta_between_norm;
                out_theta_reorient_norm(rows)=th_re_i_norm;

                % NEW
                out_speed_pre_i(rows)   = ev_speed_i_pre(ee);
                out_speed_pre_j(rows)   = ev_speed_j_pre(ee);
                out_speed_pre_ref(rows) = ev_speed_i_pre(ee);
                out_speed_pre_opp(rows) = ev_speed_j_pre(ee);
                
                out_vx_pre_i(rows)   = ev_vx_i_pre(ee);
                out_vy_pre_i(rows)   = ev_vy_i_pre(ee);
                out_vx_pre_j(rows)   = ev_vx_j_pre(ee);
                out_vy_pre_j(rows)   = ev_vy_j_pre(ee);
                
                out_vx_pre_ref(rows) = ev_vx_i_pre(ee);
                out_vy_pre_ref(rows) = ev_vy_i_pre(ee);
                out_vx_pre_opp(rows) = ev_vx_j_pre(ee);
                out_vy_pre_opp(rows) = ev_vy_j_pre(ee);

                rows = rows + 1; 
                if rows>capOut, rows=capOut; break; end
                out_pairType(rows)=ev_pairType(ee);
                out_step0(rows)=double(ev_step0(ee));
                out_stepEval(rows)=step;
                out_i(rows)=int32(i); 
                out_j(rows)=int32(j);
                out_ref(rows)="j";
                out_theta_pre_between(rows)=theta_pre_between;
                out_theta_between(rows)=theta_between;
                out_theta_reorient(rows)=th_re_j;
                out_theta_between_norm(rows)=theta_between_norm;
                out_theta_reorient_norm(rows)=th_re_j_norm;

                out_speed_pre_i(rows)   = ev_speed_i_pre(ee);
                out_speed_pre_j(rows)   = ev_speed_j_pre(ee);
                out_speed_pre_ref(rows) = ev_speed_j_pre(ee);
                out_speed_pre_opp(rows) = ev_speed_i_pre(ee);

                out_vx_pre_i(rows)   = ev_vx_i_pre(ee);
                out_vy_pre_i(rows)   = ev_vy_i_pre(ee);
                out_vx_pre_j(rows)   = ev_vx_j_pre(ee);
                out_vy_pre_j(rows)   = ev_vy_j_pre(ee);

                out_vx_pre_ref(rows) = ev_vx_j_pre(ee);
                out_vy_pre_ref(rows) = ev_vy_j_pre(ee);
                out_vx_pre_opp(rows) = ev_vx_i_pre(ee);
                out_vy_pre_opp(rows) = ev_vy_i_pre(ee);



            end
        end
    end

    % ---- MP4 frame ----
    if S.make_mp4 && mod(step, S.mp4_every)==0

        % decay markers (per saved frame)
        if ~isempty(mark_ttl)
            mark_ttl = mark_ttl - 1;
            keep = (mark_ttl > 0);
            mark_ttl = mark_ttl(keep);
            mark_pos = mark_pos(keep,:);
        end

        clf; hold on;
        axis([xmin xmax ymin ymax]); axis equal;
        set(gca,'YDir','normal'); box on;

        isR = (types==1);
        scatter(pos(~isR,1), pos(~isR,2), 45, 'filled', ...
            'MarkerFaceColor',[0.0 0.81 0.82], 'MarkerFaceAlpha',0.55);
        scatter(pos(isR,1),  pos(isR,2),  45, 'filled', ...
            'MarkerFaceColor',[1.00 0.7 0.0], 'MarkerFaceAlpha',0.55);

        stuck = stick_left > 0;
        if any(stuck)
            scatter(pos(stuck,1), pos(stuck,2), 55, 'o', ...
                'MarkerEdgeColor','k','LineWidth',1.2);
        end

        if S.show_coll_mark && ~isempty(mark_pos)
            scatter(mark_pos(:,1), mark_pos(:,2), S.coll_mark_size, 'o', ...
                'MarkerEdgeColor','k','LineWidth',1.2);
        end

        title(sprintf('step=%d  (RR stick=%d)', step, nnz(stick_left>0)));
        drawnow;

        fr = getframe(gcf);
        writeVideo(vobj, fr);
    end

end % step loop

if S.make_mp4
    close(vobj);
    disp("Saved: " + S.mp4_path);
end

%% ===================== SAVE ANGLE TABLE + SCATTER =====================
if rows == 0
    disp("No valid events logged.");
    return;
end

T = table( ...
    out_pairType(1:rows), ...
    out_step0(1:rows), ...
    out_stepEval(1:rows), ...
    out_i(1:rows), ...
    out_j(1:rows), ...
    out_ref(1:rows), ...
    out_theta_pre_between(1:rows), ...
    out_theta_between(1:rows), ...
    out_theta_reorient(1:rows), ...
    out_theta_between_norm(1:rows), ...
    out_theta_reorient_norm(1:rows), ...
    out_speed_pre_i(1:rows), ...
    out_speed_pre_j(1:rows), ...
    out_speed_pre_ref(1:rows), ...
    out_speed_pre_opp(1:rows), ...
    out_vx_pre_i(1:rows), ...
    out_vy_pre_i(1:rows), ...
    out_vx_pre_j(1:rows), ...
    out_vy_pre_j(1:rows), ...
    out_vx_pre_ref(1:rows), ...
    out_vy_pre_ref(1:rows), ...
    out_vx_pre_opp(1:rows), ...
    out_vy_pre_opp(1:rows), ...
    'VariableNames', { ...
        'pairType','step0','step_eval','i','j','ref', ...
        'theta_pre_between_deg', ...
        'theta_between_deg', ...
        'theta_reorient_deg', ...
        'theta_between_norm', ...
        'theta_reorient_norm', ...
        'speed_pre_i', ...
        'speed_pre_j', ...
        'speed_pre_ref', ...
        'speed_pre_opp', ...
        'vx_pre_i', ...
        'vy_pre_i', ...
        'vx_pre_j', ...
        'vy_pre_j', ...
        'vx_pre_ref', ...
        'vy_pre_ref', ...
        'vx_pre_opp', ...
        'vy_pre_opp' ...
    });

writetable(T, S.out_csv);
disp("Saved: " + S.out_csv);

% ===== choose what to plot =====
% raw version:
% x = T.theta_reorient_deg;
% y = T.theta_between_deg;
%
% normalized version:
x = T.theta_reorient_norm;
y = T.theta_between_norm;

mRR = (T.pairType=="RR");
mGG = (T.pairType=="GG");
mRG = (T.pairType=="RG");

figure('Color','w'); hold on;
scatter(x(mRR), y(mRR), 10, 'filled', 'MarkerFaceAlpha', 0.18);
scatter(x(mGG), y(mGG), 10, 'filled', 'MarkerFaceAlpha', 0.18);
scatter(x(mRG), y(mRG), 10, 'filled', 'MarkerFaceAlpha', 0.18);
xlabel('\theta_{reorient} / \theta_{pre,between}');
ylabel('\theta_{between} / \theta_{pre,between}');
title(sprintf('Normalized by pre-between | p_{red}=%.2f | delay=%d | tau=%.1f | sigma=%.2f', ...
    S.p_red, S.delay_steps, S.tau_turn, S.sigma_turn));
legend({'RR (stick+OU release)','GG (elastic)','RG (elastic)'}, 'Location','best');
grid on; box on;
exportgraphics(gcf, S.out_fig, 'Resolution', 200);
disp("Saved: " + S.out_fig);

%% ===================== FUNCTIONS =====================

function th = angle_between_vec(a, b, epsn)
na = norm(a); 
nb = norm(b);
if na < epsn || nb < epsn
    th = NaN; 
    return;
end
ca = dot(a,b) / (na*nb);
ca = max(-1, min(1, ca));
th = acos(ca) * 180/pi;
end

function [theta, speed, omega, stick_left, stick_partner, rr_cooldown] = enforce_rr_stick_and_release_OU( ...
    theta, speed, omega, stick_left, stick_partner, rr_cooldown, S)

idx = find(stick_left > 0);
if isempty(idx), return; end

visited = false(size(stick_left));
for aa = 1:numel(idx)
    i = idx(aa);
    if visited(i), continue; end
    j = double(stick_partner(i));
    if j <= 0 || j > numel(stick_left), continue; end

    visited(i) = true;
    visited(j) = true;

    theta(j) = theta(i);
    speed(j) = speed(i);
    omega(j) = omega(i);

    stick_left(i) = stick_left(i) - 1;
    stick_left(j) = stick_left(j) - 1;

    if stick_left(i) <= 0 || stick_left(j) <= 0
        stick_left(i) = 0; 
        stick_left(j) = 0;
        stick_partner(i) = 0; 
        stick_partner(j) = 0;

        if S.RR_release_mode == "random"
            if S.RR_release_speed == "reset"
                speed(i) = S.v0; 
                speed(j) = S.v0;
            end
            theta(i) = 2*pi*rand;
            theta(j) = 2*pi*rand;
            omega(i) = 0; 
            omega(j) = 0;
        else
            if S.RR_release_speed == "reset"
                speed(i) = S.v0; 
                speed(j) = S.v0;
            end
            theta(i) = theta(i) + S.RR_release_kick_sigma*randn;
            theta(j) = theta(j) + S.RR_release_kick_sigma*randn;
            omega(i) = omega(i) + 0.1*S.RR_release_kick_sigma*randn;
            omega(j) = omega(j) + 0.1*S.RR_release_kick_sigma*randn;
        end

        rr_cooldown(i,j) = uint16(S.RR_cooldown_steps);
        rr_cooldown(j,i) = uint16(S.RR_cooldown_steps);
    end
end
end

function [pos, theta] = reflect_walls_theta(pos, theta, S)
xmin=S.box(1); xmax=S.box(2); ymin=S.box(3); ymax=S.box(4);

vx = cos(theta);
vy = sin(theta);

hitL = (pos(:,1) < xmin + S.r);
hitR = (pos(:,1) > xmax - S.r);
if any(hitL)
    pos(hitL,1) = xmin + S.r + (xmin + S.r - pos(hitL,1));
    vx(hitL) = -vx(hitL);
end
if any(hitR)
    pos(hitR,1) = xmax - S.r - (pos(hitR,1) - (xmax - S.r));
    vx(hitR) = -vx(hitR);
end

hitB = (pos(:,2) < ymin + S.r);
hitT = (pos(:,2) > ymax - S.r);
if any(hitB)
    pos(hitB,2) = ymin + S.r + (ymin + S.r - pos(hitB,2));
    vy(hitB) = -vy(hitB);
end
if any(hitT)
    pos(hitT,2) = ymax - S.r - (pos(hitT,2) - (ymax - S.r));
    vy(hitT) = -vy(hitT);
end

theta = atan2(vy, vx);
end

function [vi_new, vj_new] = elastic_impulse_equalmass(vi, vj, n)
vrel = vi - vj;
vn = dot(vrel, n);
j_imp = -(1 + 1.0) * vn / 2;  % equal mass, e = 1
J = j_imp * n;
vi_new = vi + J;
vj_new = vj - J;
end

function [th, sp] = vec_to_theta_speed(v, epsn)
sp = max(norm(v), epsn);
th = atan2(v(2), v(1));
end

function pairs = grid_pairs_box(pos, cellSize, r, box)
% Unique pairs within rcut=2r using grid
xmin=box(1); xmax=box(2); ymin=box(3); ymax=box(4);
N = size(pos,1);
if N<2
    pairs=zeros(0,2); 
    return;
end

nx = max(1, ceil((xmax-xmin)/cellSize));
ny = max(1, ceil((ymax-ymin)/cellSize));

ix = floor((pos(:,1)-xmin)/cellSize) + 1;
iy = floor((pos(:,2)-ymin)/cellSize) + 1;
ix = min(max(ix,1), nx);
iy = min(max(iy,1), ny);

cellId = ix + (iy-1)*nx;

[sortedCell, order] = sort(cellId);
posS = pos(order,:);
orderS = order;

edges = [1; find(diff(sortedCell)~=0)+1; N+1];
cells = sortedCell(edges(1:end-1));

pairsCap = max(20000, 10*N);
pairs = zeros(pairsCap,2);
pcount = 0;

offs = [-1 -1; 0 -1; 1 -1; -1 0; 0 0; 1 0; -1 1; 0 1; 1 1];

cell_to_idx = containers.Map('KeyType','int32','ValueType','int32');
for k=1:numel(cells)
    cell_to_idx(int32(cells(k))) = int32(k);
end

rcut = 2*r + 1e-12;

for cidx=1:numel(cells)
    c = cells(cidx);
    a0 = edges(cidx);
    a1 = edges(cidx+1)-1;
    idxA = a0:a1;

    cx = mod(c-1, nx) + 1;
    cy = floor((c-1)/nx) + 1;

    for oo=1:size(offs,1)
        cx2 = cx + offs(oo,1);
        cy2 = cy + offs(oo,2);
        if cx2<1 || cx2>nx || cy2<1 || cy2>ny, continue; end
        c2 = cx2 + (cy2-1)*nx;
        if c2 < c, continue; end

        key = int32(c2);
        if ~isKey(cell_to_idx, key), continue; end
        jpos = double(cell_to_idx(key));

        b0 = edges(jpos);
        b1 = edges(jpos+1)-1;
        idxB = b0:b1;

        if c2 == c
            for ii=1:numel(idxA)-1
                iS = idxA(ii);
                pi = posS(iS,:);
                jj = idxA(ii+1:end);
                d = hypot(posS(jj,1)-pi(1), posS(jj,2)-pi(2));
                hit = jj(d < rcut);
                for hh=1:numel(hit)
                    pcount = pcount + 1;
                    if pcount > size(pairs,1)
                        pairs = [pairs; zeros(pairsCap,2)]; %#ok<AGROW>
                    end
                    pairs(pcount,:) = [orderS(iS), orderS(hit(hh))];
                end
            end
        else
            for ii=1:numel(idxA)
                iS = idxA(ii);
                pi = posS(iS,:);
                d = hypot(posS(idxB,1)-pi(1), posS(idxB,2)-pi(2));
                hit = idxB(d < rcut);
                for hh=1:numel(hit)
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
