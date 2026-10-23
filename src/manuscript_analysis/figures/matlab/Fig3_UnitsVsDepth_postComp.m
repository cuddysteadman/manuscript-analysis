%% UnitsVsDepth_postComp_plotOnly.m
% Plot-only post-processing for unit depth + waveforms + ISI
%
% LAST UPDATE: 2026-01-02
% Comm. Bio. 2026: Fig 3
%
% Loads: batch_units_depth_struct.mat (T, ALL)
% Produces (no recomputation from .dat):
%   1) Collapsed waveform heatmap (16 channels, 0..300 um, boundary=0 at top)
%   2) Unit depth histogram (units vs depth)
%   3) Waveform stack plots (16 channels) for either:
%         - manual GlobalID list/range, OR
%         - auto representative unit per unique depth (max PeakVoltage_uV)
%      with highlights:
%         - maxWaveformCh (red)
%         - boundary channel (green)
%      and caption of signed peak voltage (best channel).
%   4) Optional ISI histogram per plotted unit by loading:
%         <root>\<EXP*>\kilosort4_auto_final\s.mat  (fields: clu, res)
%
% Notes:
%   - This script assumes ALL.chList contains the 16-channel list for the unit's shank.
%   - ISI assumes res is in samples at fs_res.
%   - Compatible with older MATLAB: uses set(gca,'CLim',...) not clim().
%   - Input is read from input/ in the project (get_in_path.m). Outputs go to
%     output/figures/<id>/ in the project (3A heatmap + histogram, 3B waveforms, 3C ISI
%     histograms) as PNG/SVG/PDF with and without labels plus CSV source data
%     (get_out_path.m, save_label_variants.m).

clear; clc;
addpath(fileparts(mfilename('fullpath')));  % get_in_path.m, get_out_path.m, save_label_variants.m

%% ========================= User settings =========================
in_dir = get_in_path();
mat_name = 'batch_units_depth_struct.mat'; % required file that stores single unit data

% Output toggles
do_heatmap   = true;
do_histogram = true;
do_waveforms = true;

% ---------------- Manual exclusions / thresholds ----------------
bad_units = [18,25,115,117,233,235,236,310,311,313,315,316,304,319,322,325,328,362,365,367,380,382,385,401,420,444,469,470,493,501,504,523,526,530,535,537,539,545,546,568,573,648,656,697,687,688,693,698,717,736,740,741,757,761,762,797,875,883,904,948,987,1024,1026,1027,1138, 1158,1258,1249,1309,1317,1336,1337,1338,1358,1359,1415];                 % GlobalIDs to EXCLUDE from heatmap + scatter (e.g., [17 42 88])
peak_uV_threshold = 30;         % uV. Scatter boundary extent: last channels above COM with |peak| >= threshold

% Heatmap settings
yStep_um = 20;
depth_span_um = 300;              % show 0..300 um => 16 bins at 20 um step
unit_col_repeat = 6;              % widen each unit column (visual only)
use_grayscale = true;

% Heatmap intensity options
normalize_each_unit = false;       % true: best-channel peak normalized to 1
use_global_clim = true;           % only used if normalize_each_unit == false
global_max_uV = 100;               % [] -> auto from data; or set (e.g., 300) to clamp

% Heatmap background mode:
%   false -> unused bins set to 0 (maps to green at 0 uV)
%   true  -> unused bins kept as NaN and shown as white background
heatmap_nan_background = false;

% Heatmap color clamp (uV). Set to 100 for +/-100 uV, or [] to auto from data.
heatmap_clamp_uV = 100;

% Heatmap figure width control (helps when many units * repeat columns exceed screen pixels)
auto_fig_width = true;
px_per_col = 1;              % exported pixels per heatmap column (1 is a good default)
max_fig_width_px = 8000;     % cap to avoid extremely huge figures


% Histogram styling
hist_face = [0.35 0.35 0.35];     % dark gray fill
hist_edge = 'k';                  % black edge
hist_edge_alpha = 0.25;           % soft edge
hist_linewidth = 0.75;

% Waveform plot mode
%   "manual" -> plot manual_global_ids
%   "auto"   -> plot representative unit per unique depth (max PeakVoltage_uV)
waveform_mode = "manual";           % "manual" or "auto"
manual_global_ids = [24, 55, 108, 312, 366, 695,837,989,1148,1218,1283,1339,1364,1404,1423];   % used only in manual mode
max_units_to_plot = inf;          % cap manual list if desired

% Waveform plot quality + layout (KEEP fig_w = 200 per your request)
png_dpi = 600;
fig_w = 200;     % intentional (narrow)
fig_h = 780;

% Waveform y-limits (um) relative to boundary depth=0
Y_ABOVE_SURFACE_UM = 100;         % show negative depths up to -300
Y_BELOW_SURFACE_UM = 400;         % show positive depths down to +300

% Waveform vertical scaling
targetHalfSpan_um = 5 * yStep_um; % increase to make waveforms taller
use_global_waveform_scale = true;
global_scale_percentile = 99;     % 95-99 typical; 100 = absolute max

% ISI plots
do_plot_isi = true;
ks_root_dir = "D:\Nov2024_Merck";       % root containing EXP* folders

ks_subdir = "kilosort4_auto_final";      % default spike-sort folder inside EXP folder

fs_res = 20000;                          % res sample rate (Hz) -- set correctly
isi_xlim_ms = 200;                       % x-axis range (ms)
isi_bin_ms = 1;                          % bin width (ms)

%% ========================= Load data =========================
mat_path = fullfile(in_dir, mat_name);
if ~exist(mat_path, 'file')
    error('Cannot find %s in %s', mat_name, in_dir);
end

S = load(mat_path);
if ~isfield(S,'T') || ~isfield(S,'ALL')
    error('MAT file missing T and/or ALL. Found fields: %s', strjoin(fieldnames(S), ', '));
end
T = S.T;
ALL = S.ALL;

fprintf('Loaded %d units from %s\n', height(T), mat_path);

% Basic validation
reqALL = {'wfCell','chList','boundaryIdx','maxWaveformCh1'};
for ii = 1:numel(reqALL)
    if ~isfield(ALL, reqALL{ii})
        error('ALL is missing required field: %s', reqALL{ii});
    end
end

% Ensure filebase is string for easier matching
if iscell(T.filebase)
    T.filebase = string(T.filebase);
elseif ischar(T.filebase)
    T.filebase = string(cellstr(T.filebase));
elseif ~isstring(T.filebase)
    T.filebase = string(T.filebase);
end

%% ========================= Heatmap + depth scatter (y_COM) =========================
if do_heatmap
    % This section provides TWO panels:
    %   (A) Original collapsed waveform heatmap (grayscale)
    %   (B) Scatter plot of unit COM depth from y_COM_um, with polarity colors + boundary dots
    %
    % Updates:
    %   1) Users can exclude units by GlobalID via bad_units = [];
    %   2) Scatter uses y_COM_um (negative) -> depth_um = -y_COM_um (positive)
    %   3) Boundary dots: for each unit, find channel extent around COM where peak drops below threshold
    %      (default peak_uV_threshold = 10 uV). Boundary dots are gray.

    % ---- Required table fields ----
    if ~ismember('GlobalID', T.Properties.VariableNames)
        error('T is missing required field: GlobalID');
    end
    if ~ismember('y_COM_um', T.Properties.VariableNames)
        error('T is missing required field: y_COM_um (saved by the computation script).');
    end

    % Polarity field (robust to naming)
    pol_field = "";
    if ismember('Polarity', T.Properties.VariableNames)
        pol_field = "Polarity";
    elseif ismember('polarity', T.Properties.VariableNames)
        pol_field = "polarity";
    end
    if pol_field == ""
        warning('T does not contain Polarity/polarity; COM dots will be gray.');
    end

    % ---- Exclude units (by GlobalID) ----
    keep_mask = true(height(T),1);
    if exist('bad_units','var') && ~isempty(bad_units)
        keep_mask = ~ismember(T.GlobalID, bad_units(:));
    end
    T_h = T(keep_mask, :);

% Subselect ALL to match (minus exclusions), then create TWO different orderings:
%   - Heatmap order: original behavior (sorted by discrete computed Depth_um)
%   - Scatter order: sorted by COM depth (-y_COM_um)
%
% NOTE: We keep the two sequences intentionally different per your request.

% --- Shallow copy of ALL restricted to kept units (same row alignment as T_h) ---
ALL_h = ALL;
try
    ALL_h.wfCell         = ALL.wfCell(keep_mask);
    ALL_h.chList         = ALL.chList(keep_mask);
    ALL_h.boundaryIdx    = ALL.boundaryIdx(keep_mask);
    ALL_h.maxWaveformCh1 = ALL.maxWaveformCh1(keep_mask);
    % Optional precomputed per-channel peaks (if present)
    if isfield(ALL, 'peakAbs16_uV'), ALL_h.peakAbs16_uV = ALL.peakAbs16_uV(keep_mask); end
    if isfield(ALL, 'peakAbs_uV'),   ALL_h.peakAbs_uV   = ALL.peakAbs_uV(keep_mask);   end
    if isfield(ALL, 'peak16_uV'),    ALL_h.peak16_uV    = ALL.peak16_uV(keep_mask);    end
catch
    % If ALL isn't aligned or subselect fails, heatmap/scatter may error later.
end

nU_all = height(T);
nU_keep = height(T_h);

% ---- Heatmap ordering (Depth_um discrete steps) ----
if ~ismember('Depth_um', T_h.Properties.VariableNames)
    error('T is missing required field: Depth_um (needed to sort heatmap units by computed depth).');
end
[~, ord_heat] = sort(double(T_h.Depth_um), 'ascend');   % shallow->deep
T_heat = T_h(ord_heat, :);
ALL_heat = ALL_h;
try
    ALL_heat.wfCell         = ALL_h.wfCell(ord_heat);
    ALL_heat.chList         = ALL_h.chList(ord_heat);
    ALL_heat.boundaryIdx    = ALL_h.boundaryIdx(ord_heat);
    ALL_heat.maxWaveformCh1 = ALL_h.maxWaveformCh1(ord_heat);
    if isfield(ALL_sc, 'peakAbs16_uV'), ALL_heat.peakAbs16_uV = ALL_h.peakAbs16_uV(ord_heat); end
    if isfield(ALL_sc, 'peakAbs_uV'),   ALL_heat.peakAbs_uV   = ALL_h.peakAbs_uV(ord_heat);   end
    if isfield(ALL_sc, 'peak16_uV'),    ALL_heat.peak16_uV    = ALL_h.peak16_uV(ord_heat);    end
catch
end

% ---- Scatter ordering (COM depth) ----
depth_um_COM_all = -double(T_h.y_COM_um);               % positive depth
[~, ord_sc] = sort(depth_um_COM_all, 'ascend', 'MissingPlacement','last');
T_sc = T_h(ord_sc, :);
ALL_sc = ALL_h;
try
    ALL_h.wfCell         = ALL_h.wfCell(ord_sc);
    ALL_h.chList         = ALL_h.chList(ord_sc);
    ALL_h.boundaryIdx    = ALL_h.boundaryIdx(ord_sc);
    ALL_h.maxWaveformCh1 = ALL_h.maxWaveformCh1(ord_sc);
    if isfield(ALL_sc, 'peakAbs16_uV'), ALL_h.peakAbs16_uV = ALL_h.peakAbs16_uV(ord_sc); end
    if isfield(ALL_sc, 'peakAbs_uV'),   ALL_h.peakAbs_uV   = ALL_h.peakAbs_uV(ord_sc);   end
    if isfield(ALL_sc, 'peak16_uV'),    ALL_h.peak16_uV    = ALL_h.peak16_uV(ord_sc);    end
catch
end

nU_heat = height(T_heat);
nU_sc   = height(T_sc);

% ---- Shared axes settings ----
depth_axis = (0:yStep_um:depth_span_um).';
nD = numel(depth_axis);
ucr = max(1, round(unit_col_repeat));

% =====================================================================
% (A) Original collapsed waveform heatmap (same as before; grayscale)
% =====================================================================
    if heatmap_nan_background
        M = nan(nD, nU_heat);      % keep NaNs
    else
        M = zeros(nD, nU_heat);    % fill unused as 0 uV
    end

    for u = 1:nU_heat
        wf     = ALL_heat.wfCell{u};
        chList = ALL_heat.chList{u};
        bIdx   = ALL_heat.boundaryIdx(u);
        hiCh   = ALL_heat.maxWaveformCh1(u);

        if isempty(wf) || isempty(chList) || numel(chList) ~= 16 || isempty(bIdx) || ~isfinite(bIdx) || bIdx < 1 || bIdx > 16
            continue;
        end

        % Use best-channel peak sample to define the "snapshot" amplitude across channels (original behavior)
        wbest = double(wf(hiCh,:));
        [bestAbs, peak_i] = max(abs(wbest));
        if bestAbs <= 0, bestAbs = 1; end

        raw16 = nan(16,1);
        for jj = 1:16
            ch = chList(jj);
            % SIGNED sample at the best-channel peak time (blue=negative, red=positive)
            raw16(jj) = double(wf(ch, peak_i));
        end

        if normalize_each_unit
            val16 = raw16 / bestAbs;
        else
            val16 = raw16;
        end

        depth_rel = (bIdx - (1:16)) * yStep_um;  % 0 at boundary, deeper positive (kept as before)

        for jj = 1:16
            d = depth_rel(jj);
            if d < 0 || d > depth_span_um
                continue;
            end
            row = round(d / yStep_um) + 1;
            if row >= 1 && row <= nD
                M(row, u) = val16(jj);
            end
        end
    end

    if ucr > 1
        Mplot = repelem(M, 1, ucr);
    else
        Mplot = M;
    end

    % ---- Figure width: optionally scale with number of heatmap columns ----
    baseW = 1400; baseH = 980;
    if exist('auto_fig_width','var') && auto_fig_width
        targetW = round(size(Mplot,2) * px_per_col);
        W = min(max(baseW, targetW), max_fig_width_px);
    else
        W = baseW;
    end
    fig = figure('Color','w','Position',[120 80 W baseH]);

    ax1 = subplot(2,1,1);
    imagesc(ax1, Mplot);
    axis(ax1,'tight');
    set(ax1,'YDir','reverse'); % 0 at TOP, deeper at BOTTOM

    colormap(ax1, myjet(256));
    cb = colorbar(ax1);

    % Clamp display range (centered at 0)
    if exist('heatmap_clamp_uV','var') && ~isempty(heatmap_clamp_uV) && isfinite(heatmap_clamp_uV)
        clim = [-abs(heatmap_clamp_uV) abs(heatmap_clamp_uV)];
    else
        g = max(abs(Mplot(:)), [], 'omitnan');
        if isempty(g) || ~isfinite(g) || g <= 0
            g = 1;
        end
        clim = [-g g];
    end
    set(ax1,'CLim',clim);

    % NaN background toggle: NaNs are white; real zeros map to mid-color (green)
    hImg = findobj(ax1,'Type','image');
    if heatmap_nan_background
        if ~isempty(hImg), set(hImg(1), 'AlphaData', ~isnan(Mplot)); end
        set(ax1,'Color',[1 1 1]); % white background behind transparent NaNs
    else
        if ~isempty(hImg), set(hImg(1), 'AlphaData', 1); end
        set(ax1,'Color',[1 1 1]);
    end

    if normalize_each_unit
        ylabel(cb, 'Normalized V (uV) at best-channel peak sample');
    else
        ylabel(cb, 'V (uV) at best-channel peak sample');
    end

    set(ax1, 'YTick', 1:nD, 'YTickLabel', string(depth_axis));
    ylabel(ax1,'Depth from surface (\mum)');

    % X ticks as GlobalID (not repeated columns). NOTE: order preserved from T (minus excluded units).
    if ucr > 1
        centers_heat = (1:nU_heat) * ucr - (ucr-1)/2;
    else
        centers_heat = 1:nU_heat;
    end
%     tick_step = max(1, round(nU_heat/12));
%     tick_units = 1:tick_step:nU_heat;
%     set(ax1, 'XTick', centers_heat(tick_units), 'XTickLabel', string(T_heat.GlobalID(tick_units)));
%     xlabel(ax1,'Global Unit ID');

    % Clamp display range to +/-100 uV around 0
    % set(ax1, 'CLim', [-100 100]);
    set(ax1, 'CLim', [-50 50]);

    title(ax1, sprintf('Collapsed waveform heatmap | kept=%d/%d | repeat=%d | normalize=%d', nU_heat, nU_all, ucr, normalize_each_unit), 'Interpreter','none');

    % =====================================================================
    % (B) Scatter plot: COM depth from y_COM_um + boundary dots
    % =====================================================================
    ax2 = subplot(2,1,2); hold(ax2,'on');

    % Default boundary threshold
    if ~exist('peak_uV_threshold','var') || isempty(peak_uV_threshold)
        peak_uV_threshold = 10; % uV
    end

    % X positions for SCATTER: separate unit order (sorted by COM depth)
    if ucr > 1
        centers_sc = (1:nU_sc) * ucr - (ucr-1)/2;
    else
        centers_sc = 1:nU_sc;
    end

    % Convert y_COM_um (negative) to positive depth (um)
    depth_um_COM = -double(T_sc.y_COM_um);

    % Keep only in-range COM points
    in_range = isfinite(depth_um_COM) & depth_um_COM >= 0 & depth_um_COM <= depth_span_um;
    depth_um_COM_plot = depth_um_COM(in_range);

    % X positions aligned with heatmap unit centers
    x_all = centers_sc; % 1 x nU
    x_plot = x_all(in_range);

    % Polarity colors (COM dots): blue if polarity > 0, else red
    if pol_field ~= ""
        pol = double(T_sc.(pol_field));
        pol = pol(in_range);
        is_pos = pol > 0;

        scatter(ax2, x_plot(is_pos),  depth_um_COM_plot(is_pos),  10, 'b', 'filled', 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
        scatter(ax2, x_plot(~is_pos), depth_um_COM_plot(~is_pos), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
    else
        scatter(ax2, x_plot, depth_um_COM_plot, 10, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
    end

    % Boundary dots (gray): extent around COM where per-channel peak >= threshold
    xb = nan(nU_sc, 2);  % [above below] x coords
    yb = nan(nU_sc, 2);  % [above below] depths (um)

    for u = 1:nU_sc
        if ~in_range(u), continue; end % only if COM is in range

        wf     = ALL_h.wfCell{u};
        chList = ALL_h.chList{u};
        bIdx   = ALL_h.boundaryIdx(u);

        if isempty(wf) || isempty(chList) || numel(chList) ~= 16 || isempty(bIdx) || ~isfinite(bIdx) || bIdx < 1 || bIdx > 16
            continue;
        end

        % --- get per-channel peak abs (uV) across waveform time ---
        peakAbs16 = [];
        if isfield(ALL_sc, 'peakAbs16_uV')
            try, peakAbs16 = double(ALL_h.peakAbs16_uV{u}); catch, end
        end
        if isempty(peakAbs16) && isfield(ALL_sc, 'peak16_uV')
            try, peakAbs16 = abs(double(ALL_h.peak16_uV{u})); catch, end
        end
        if isempty(peakAbs16) && isfield(ALL_sc, 'peakAbs_uV')
            try, peakAbs16 = double(ALL_h.peakAbs_uV{u}); catch, end
        end
        if isempty(peakAbs16) || numel(peakAbs16) ~= 16
            peakAbs16 = nan(16,1);
            for jj = 1:16
                ch = chList(jj);
                peakAbs16(jj) = max(abs(double(wf(ch,:))));
            end
        else
            peakAbs16 = peakAbs16(:);
        end

        % Map COM depth (positive) to a channel index jj around the boundary index (kept consistent with heatmap)
        dcom = depth_um_COM(u);
        jjCOM = round(bIdx - dcom / yStep_um);
        jjCOM = max(1, min(16, jjCOM));

        if ~isfinite(peakAbs16(jjCOM)) || peakAbs16(jjCOM) < peak_uV_threshold
            continue;
        end

        % Search "above" (shallower direction; increasing jj) until drop below threshold
        jj = jjCOM;
        while jj < 16 && isfinite(peakAbs16(jj+1)) && peakAbs16(jj+1) >= peak_uV_threshold
            jj = jj + 1;
        end
        jj_above = jj;

        % Search "below" (deeper direction; decreasing jj) until drop below threshold
        jj = jjCOM;
        while jj > 1 && isfinite(peakAbs16(jj-1)) && peakAbs16(jj-1) >= peak_uV_threshold
            jj = jj - 1;
        end
        jj_below = jj;

        d_above = (bIdx - jj_above) * yStep_um;
        d_below = (bIdx - jj_below) * yStep_um;

        if d_above >= 0 && d_above <= depth_span_um
            xb(u,1) = centers_sc(u);
            yb(u,1) = d_above;
        end
        if d_below >= 0 && d_below <= depth_span_um
            xb(u,2) = centers_sc(u);
            yb(u,2) = d_below;
        end
    end

    scatter(ax2, xb(:), yb(:), 9, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.45, 'MarkerEdgeAlpha', 0.45);

    set(ax2,'YDir','reverse');
    set(ax2, 'YTick', depth_axis, 'YTickLabel', string(depth_axis));
    ylabel(ax2,'Depth from surface (\mum)');
    xlabel(ax2,'Global Unit ID');

% X tick labels for scatter order (sorted by COM depth)
tick_step_sc = max(1, round(nU_sc/12));
tick_units_sc = 1:tick_step_sc:nU_sc;
set(ax2, 'XTick', centers_sc(tick_units_sc), 'XTickLabel', string(T_sc.GlobalID(tick_units_sc)));
xlim(ax2, [min(centers_sc)-ucr, max(centers_sc)+ucr]);
    ylim(ax2, [-yStep_um/2, depth_span_um + yStep_um/2]);
    box(ax2,'off');

    title(ax2, sprintf('COM depth from y\\_COM\\_um (blue=pol>0, red=pol<=0) + boundaries (|peak| \\ge %g uV)', peak_uV_threshold), 'Interpreter','none');

    save_label_variants(fig, get_out_path('3A'), '3A_unit_depth_heatmap');
end
%% ========================= Depth histogram =========================
if do_histogram
    % Exclude manual bad units (by GlobalID)
    if exist('bad_units','var') && ~isempty(bad_units)
        good_mask_hist = ~ismember(T.GlobalID, bad_units(:));
    else
        good_mask_hist = true(height(T),1);
    end

    depth_um_units = T.Depth_um(good_mask_hist);

    bin_edges = (-yStep_um/2) : yStep_um : (depth_span_um + yStep_um/2);
    bin_centers = 0:yStep_um:depth_span_um;

    counts = histcounts(depth_um_units, bin_edges);

    fig = figure('Color','w','Position',[120 80 320 520]);
    ax = axes(fig);

    bh = barh(ax, bin_centers, counts, 1.0);
    bh.FaceColor = hist_face;
    bh.EdgeColor = hist_edge;
    bh.LineWidth = hist_linewidth;
    try
        bh.EdgeAlpha = hist_edge_alpha;
    catch
    end

    set(ax,'YDir','reverse'); % 0 at top
    xlabel(ax,'Unit count');
    ylabel(ax,'Depth from surface (\mum)');
    set(ax,'YTick', bin_centers);
    ylim(ax, [-yStep_um/2, depth_span_um + yStep_um/2]);
    title(ax, sprintf('Unit depth distribution (N=%d)', numel(depth_um_units)));
    box(ax,'off');

    out_3A = get_out_path('3A');
    writetable(table(bin_centers(:), counts(:), 'VariableNames', {'Depth_um', 'UnitCount'}), ...
        fullfile(out_3A, '3A_unit_depth_histogram.csv'));
    save_label_variants(fig, out_3A, '3A_unit_depth_histogram');
end

%% ========================= Waveform + optional ISI =========================
if do_waveforms
    % Global waveform scaling precompute
    if use_global_waveform_scale
        if global_scale_percentile >= 100
            globalMaxAbsWF = max(T.PeakVoltage_uV);
        else
            globalMaxAbsWF = prctile(T.PeakVoltage_uV, global_scale_percentile);
        end
        if isempty(globalMaxAbsWF) || ~isfinite(globalMaxAbsWF) || globalMaxAbsWF <= 0
            globalMaxAbsWF = 1;
        end
        scale_um_per_uV_global = targetHalfSpan_um / globalMaxAbsWF;
    else
        scale_um_per_uV_global = nan;
    end

    plot_one_unit = @(u, out_dir, prefix) plot_unit_waveform( ...
        u, out_dir, prefix, ...
        T, ALL, yStep_um, ...
        use_global_waveform_scale, scale_um_per_uV_global, targetHalfSpan_um, ...
        Y_ABOVE_SURFACE_UM, Y_BELOW_SURFACE_UM, ...
        fig_w, fig_h, png_dpi, ...
        do_plot_isi, ks_root_dir, ks_subdir, fs_res, isi_xlim_ms, isi_bin_ms);

    if waveform_mode == "manual"
        out_dir = get_out_path('3B');

        gids = manual_global_ids(:).';
        gids = unique(gids, 'stable');
        if isfinite(max_units_to_plot)
            gids = gids(1:min(numel(gids), max_units_to_plot));
        end

        fprintf('\nManual waveform mode: %d GlobalIDs -> %s\n', numel(gids), out_dir);
        n_saved = 0;
        for k = 1:numel(gids)
            gid = gids(k);
            u = find(T.GlobalID == gid, 1, 'first');
            if isempty(u)
                fprintf('GlobalID %d not found. Skipping.\n', gid);
                continue;
            end
            plot_one_unit(u, out_dir, sprintf('U%04d', gid));
            n_saved = n_saved + 1;
        end
        fprintf('Manual waveform figures saved: %d\n', n_saved);

    else
        out_dir = get_out_path('3B');

        depths = unique(T.Depth_um);
        depths = sort(depths(:), 'ascend');

        fprintf('\nAuto waveform mode: %d unique depths -> %s\n', numel(depths), out_dir);
        n_saved = 0;
        for di = 1:numel(depths)
            d0 = depths(di);
            idx = find(abs(T.Depth_um - d0) < 1e-9);
            if isempty(idx), continue; end

            [~, imax] = max(T.PeakVoltage_uV(idx));
            u = idx(imax);
            gid = T.GlobalID(u);

            plot_one_unit(u, out_dir, sprintf('rep_depth%03dum_U%04d', round(d0), gid));
            n_saved = n_saved + 1;
        end
        fprintf('Auto waveform figures saved: %d\n', n_saved);
    end
end

%% ========================= Local functions =========================
function plot_unit_waveform(u, out_dir, prefix, ...
    T, ALL, yStep_um, ...
    use_global_waveform_scale, scale_um_per_uV_global, targetHalfSpan_um, ...
    Y_ABOVE_SURFACE_UM, Y_BELOW_SURFACE_UM, ...
    fig_w, fig_h, png_dpi, ...
    do_plot_isi, ks_root_dir, ks_subdir, fs_res, isi_xlim_ms, isi_bin_ms)

    wf     = ALL.wfCell{u};
    chList = ALL.chList{u};
    bIdx   = ALL.boundaryIdx(u);
    hiCh   = ALL.maxWaveformCh1(u);

    gid = T.GlobalID(u);
    d0  = T.Depth_um(u);
    sh  = T.ShankID(u);
    clu = T.CluID(u);
    filebase = char(T.filebase(u));

    if isempty(wf) || isempty(chList) || numel(chList) ~= 16 || isempty(bIdx) || ~isfinite(bIdx) || bIdx < 1 || bIdx > 16
        fprintf('%s: U%04d invalid wf/chList/bIdx. Skipping.\n', prefix, gid);
        return;
    end
    if size(wf,1) < max(chList)
        fprintf('%s: U%04d waveform size mismatch. Skipping.\n', prefix, gid);
        return;
    end

    nSampWF = size(wf,2);
    depth_rel = (bIdx - (1:16)) * yStep_um;

    wbest = double(wf(hiCh,:));
    [~, peak_i] = max(abs(wbest));
    peak_uV_signed = wbest(peak_i);

    if use_global_waveform_scale
        scale_um_per_uV = scale_um_per_uV_global;
    else
        bestAbs = max(abs(wbest));
        if bestAbs <= 0, bestAbs = 1; end
        scale_um_per_uV = targetHalfSpan_um / bestAbs;
    end

    fig = figure('Color','w','Position',[120 80 fig_w fig_h]);
    ax = axes(fig); hold(ax,'on');

    for c = 1:16
        ch = chList(c);
        y0 = depth_rel(c);
        yy = y0 - double(wf(ch,:)) * scale_um_per_uV;

        if ch == hiCh
            plot(ax, 1:nSampWF, yy, 'r', 'LineWidth', 2.0);
        elseif c == bIdx
            plot(ax, 1:nSampWF, yy, 'g', 'LineWidth', 2.0);
        else
            plot(ax, 1:nSampWF, yy, 'k', 'LineWidth', 1.1);
        end
    end

    set(ax,'YDir','reverse');
    ylim(ax, [-Y_ABOVE_SURFACE_UM, Y_BELOW_SURFACE_UM]);
    xlim(ax,[1 nSampWF]);
    set(ax,'XTick',[]);
    ylabel(ax,'Depth from surface (\mum)');

  %  title(ax, sprintf('%s | Depth %g \mum | %s | sh %d | clu %d', ...
  %      prefix, d0, filebase, sh, clu), 'Interpreter','none');

  %  text(ax, 0.98, 0.98, sprintf('Peak(best): %+0.1f \muV', peak_uV_signed), ...
  %      'Units','normalized','HorizontalAlignment','right', ...
  %      'VerticalAlignment','top','FontWeight','bold','Color','k');

    save_label_variants(fig, out_dir, sprintf('3B_%s_%s', prefix, sanitize_filename(filebase)));

    if do_plot_isi
        plot_and_save_isi_hist(u, get_out_path('3C'), prefix, T, ks_root_dir, ks_subdir, fs_res, isi_xlim_ms, isi_bin_ms, png_dpi);
    end
end

function plot_and_save_isi_hist(u, out_dir, prefix, T, ks_root_dir, ks_subdir, fs_res, isi_xlim_ms, isi_bin_ms, png_dpi)

    filebase = char(T.filebase(u));
    cluID = double(T.CluID(u));
    gid   = double(T.GlobalID(u));

    % Extract leading "EXP###" token
    tok = regexp(filebase, '^(EXP\d+)', 'tokens', 'once', 'ignorecase');
    if isempty(tok)
        fprintf('ISI: U%04d: cannot parse EXP### from filebase="%s"\n', gid, filebase);
        return;
    end
    exp_prefix = upper(tok{1});

    % Find the actual EXP folder (because folder name is longer than EXP##)
    d = dir(fullfile(char(ks_root_dir), [exp_prefix '*']));
    d = d([d.isdir]);
    d = d(~startsWith({d.name}, '.'));

    if isempty(d)
        fprintf('ISI: U%04d: no folder found for %s under %s\n', gid, exp_prefix, ks_root_dir);
        return;
    end

    % Prefer exact/closest match to filebase if multiple hits
    if numel(d) > 1
        names = string({d.name});
        hit = contains(names, string(filebase));
        if any(hit)
            ii = find(hit, 1, 'first');
        else
            ii = 1;
        end
        exp_dir = fullfile(d(ii).folder, d(ii).name);
    else
        exp_dir = fullfile(d(1).folder, d(1).name);
    end

    s_path = fullfile(exp_dir, char(ks_subdir), 's.mat');
    if ~exist(s_path, 'file')
        fprintf('ISI: U%04d: missing s.mat: %s\n', gid, s_path);
        return;
    end

    % === Canonical format ONLY: s.clu and s.res ===
    S = load(s_path, 's');
    if ~isfield(S,'s') || ~isfield(S.s,'clu') || ~isfield(S.s,'res')
        error('ISI: s.mat must contain struct "s" with fields "clu" and "res": %s', s_path);
    end

    clu = double(S.s.clu(:));
    res = double(S.s.res(:));

    if numel(clu) ~= numel(res)
        error('ISI: clu/res length mismatch in %s', s_path);
    end

    % Select spikes for this cluster id
    mask = (clu == cluID);
    res_u = res(mask);

    if numel(res_u) < 2
        fprintf('ISI: U%04d: not enough spikes for clu %d (%d spikes)\n', gid, cluID, numel(res_u));
        return;
    end

    res_u = sort(res_u);
    isi_ms = diff(res_u) / double(fs_res) * 1000;

    % Pretty histogram (soft edge, no legend)
    fig = figure('Color','w','Position',[120 120 520 360]);
    ax = axes(fig);

    % --- ISI histogram: log10 x-axis, 0–10 s, no edges ---
    isi_s = isi_ms / 1000;        % convert ms → seconds

    edges = logspace(log10(1e-3), log10(10), 60);  % 1 ms → 10 s (log bins)

    % Save the exact histogram source data and individual ISIs to CSV.
    % Histcounts uses the same bin boundaries as histogram(ax, isi_s, edges).
    isi_counts = histcounts(isi_s, edges);
    bin_lower_s = edges(1:end-1).';
    bin_upper_s = edges(2:end).';
    bin_center_s = sqrt(bin_lower_s .* bin_upper_s); % geometric center for log bins

    histogram_table = table(bin_lower_s, bin_upper_s, bin_center_s, ...
        isi_counts(:), 'VariableNames', ...
        {'BinLower_s', 'BinUpper_s', 'BinCenter_s', 'Count'});
    raw_isi_table = table(isi_s(:), 'VariableNames', {'ISI_s'});

    out_hist_csv = fullfile(out_dir, sprintf('3C_%s_ISI_histogram.csv', prefix));
    out_raw_csv = fullfile(out_dir, sprintf('3C_%s_ISI_raw.csv', prefix));
    writetable(histogram_table, out_hist_csv);
    writetable(raw_isi_table, out_raw_csv);
    fprintf('Saved ISI data: %s\n                %s\n', out_hist_csv, out_raw_csv);

    h = histogram(ax, isi_s, edges);
    h.FaceColor = [0.635294117647059 0.0784313725490196 0.184313725490196];        % dark red fill
    h.FaceAlpha = 0.9;
    h.EdgeColor = 'none';         % no edges

    set(ax, 'XScale', 'log');
    xlim(ax, [1e-3 10]);

    box(ax, 'off');
%     xlabel(ax, 'Inter-spike interval (s)');
%     ylabel(ax, 'Count');

%     title(ax, sprintf('%s | U%04d | clu %d | n=%d', prefix, gid, cluID, numel(res_u)), ...
%         'Interpreter','none');
    
    % new style exported by manual changes
    set(ax,'FontName','Arial','FontSize',26,'LineWidth',5,...
    'MinorGridLineStyle','none','XMinorTick','on','XScale','log');

    save_label_variants(fig, out_dir, sprintf('3C_%s_ISI', prefix));
end


function s2 = sanitize_filename(s)
    s2 = regexprep(s, '[^\w\-\.]+', '_');
end
