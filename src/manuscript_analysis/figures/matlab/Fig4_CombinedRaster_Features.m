%% ============================================================
%  Combined raster (unsorted+sorted) stacked by shank + feature traces
%  Comm. Bio. 2026: Figure 4A, 4E
%  Outputs go to output/figures/4A and 4E (PNG/SVG/PDF with and without labels, plus CSV)
%  via get_out_path.m and save_label_variants.m.
%
%  Updates (Oct 15, 2026):
%   - Manual start time T0 (s): select FIRST Nspk spikes with |Amp|>=ampThresh_uV AFTER T0
%   - Combined raster in ONE axes with single x-axis:
%       For each shank: 16 unsorted channel rows (BLACK) then sorted unit rows (colored)
%       Sorted block height varies with number of active units in the window
%   - Sorted unit colors: use LINES() (high-contrast) instead of JET
%   - Skip units with zero spikes in the window (no empty rows)
%   - Feature traces: plot RAW values for the selected Nspk spikes (no z-scoring),
%       y-limits set by min/max over the selected spikes
% ============================================================

clear; clc;
addpath(fileparts(mfilename('fullpath')));  % get_out_path.m, save_label_variants.m

%% ---------------- User parameters ----------------
matFile = "EXP43_CTRL-CO-DAY150-7_2ND-REC_NOV24.spikes.cellinfo.mat";
projRoot = fileparts(fileparts(get_out_path()));   % <project root>/output/figures -> <project root>
csvFile = fullfile(projRoot, "input", "fig4_combined_raster_features.csv"); % 500-spike subset of new_processed_features.csv (manuscript)
 
fs = 20000;                 % Hz
ampThresh_uV = 25;          % suprathreshold criterion (absolute)
Nspk = 500;                 % number of suprathreshold spikes to define window
T0 = 1021;                   % manual start time (s)

% Marker sizes
ms_sorted   = 8;
ms_unsorted = 12;

% Feature histogram settings (uses the SAME selected spikes used in the trace plot)
doFeatureHists = true;     % <--- USER TOGGLE
histNBins = 30;

% Vertical spacing multiplier for UNSORTED raster rows (channels).
%   1   = standard (adjacent rows)
%   1.5 = looser spacing
%   2   = double spacing, etc.
unsortedRowSpacing = 2;   % <--- USER ADJUST

% Optional: highlight specific spike IDs (within the selected Nspk window)
%   - Provide spike IDs in [1..Nspk]. Empty = no highlighting.
%   - Highlighting applies to:
%       (i) UNSORTED raster spikes that correspond to those selected spike IDs
%      (ii) Feature trace points at those spike IDs
highlightSpikeIdx1 = [     7
     8
     9
    10
    16
    17
    18
    19
    20
   120
   121
   122
   123
   124
   125
   126
   136
   137
   138
   139
   140
   267
   268
   269
   270
   271
   296
   297
   298
   299
   300
   301
   302
   387
   388
   389
   390
   391
   396
   397
   398
   399
   400
   401];          % clu51 (UID 24)
highlightColor1    = [0.00,0.00,1.00];   % RGB

highlightSpikeIdx2 = [127 341];                 % clu82 (UID 34)
highlightColor2    = [1.00,0.00,0.00];   % RGB (set 2)


nShanks = 4;
rowsPerShank = 16;          % fixed (channels)

% Visual spacing between shank blocks (in "rows")
gapRows = 3;

% Optional: save figures
doSave = true;
doExportCsv = true;   % Export source data for combined raster and feature scatter
outDir4A = get_out_path('4A');   % output/figures/4A: combined raster
outDir4E = get_out_path('4E');   % output/figures/4E: feature traces and histograms

%% ============================================================
%  Load CSV, pick first Nspk suprathreshold spikes AFTER T0, define window
% ============================================================

opts = detectImportOptions(csvFile);
opts.VariableNamingRule = "preserve";   % keep EXACT headers
T = readtable(csvFile, opts);

% Required columns
reqCols = {'Sample','Channel','Shank','Amplitude (uV)'};
for i = 1:numel(reqCols)
    assert(any(strcmp(T.Properties.VariableNames, reqCols{i})), 'CSV missing column: %s', reqCols{i});
end

sample = double(T.("Sample"));
chan   = double(T.("Channel"));
shank_csv = double(T.("Shank")) + 1;        % CSV is 0-index -> convert to 1-index
amp_uV = double(T.("Amplitude (uV)"));

% Sort by time
[~,ord] = sort(sample,'ascend');
sample = sample(ord);
chan = chan(ord);
shank_csv = shank_csv(ord);
amp_uV = amp_uV(ord);
T = T(ord,:);

time_s = sample / fs;

% Select first Nspk spikes meeting criteria AFTER T0
idx_keep = find((time_s >= T0) & (abs(amp_uV) >= ampThresh_uV));
assert(~isempty(idx_keep), 'No spikes meet |Amplitude| >= %.1f uV after T0=%.1f s.', ampThresh_uV, T0);
assert(numel(idx_keep) >= Nspk, 'Only %d spikes meet |Amp|>=%.1f uV after T0=%.1f s (need %d).', ...
    numel(idx_keep), ampThresh_uV, T0, Nspk);

idxSel = idx_keep(1:Nspk);      % indices into the FULL sorted arrays
tStart_s = time_s(idxSel(1));
tEnd_s   = time_s(idxSel(end));

fprintf('Window by first %d spikes with |Amp|>=%.1f uV after T0=%.1f s: start=%.6f s, end=%.6f s\n', ...
    Nspk, ampThresh_uV, T0, tStart_s, tEnd_s);

%% ============================================================
%  Load sorted spikes (.mat)
% ============================================================

S = load(matFile);               % expects variable "spikes"
assert(isfield(S,'spikes'), 'MAT file missing variable "spikes".');
spikes = S.spikes;

% Unit spike times (prefer seconds)
% Common conventions:
%   spikes.ts    -> sample indices
%   spikes.times -> seconds
% We'll use spikes.times when available; otherwise convert spikes.ts using spikes.sr.

if isfield(spikes,'times') && iscell(spikes.times) && ~isempty(spikes.times)
    tcell_all = spikes.times(:);   % seconds
elseif isfield(spikes,'ts') && iscell(spikes.ts) && ~isempty(spikes.ts)
    assert(isfield(spikes,'sr') && ~isempty(spikes.sr), 'spikes.ts present but spikes.sr missing; cannot convert samples->seconds.');
    sr_sorted = double(spikes.sr);
    tcell_all = spikes.ts(:);
    % convert each unit's sample indices to seconds
    for uu = 1:numel(tcell_all)
        if ~isempty(tcell_all{uu})
            tcell_all{uu} = double(tcell_all{uu}) / sr_sorted;
        end
    end
else
    error('Sorted spikes missing spikes.times or spikes.ts (cell arrays).');
end

nUnitsTotal = numel(tcell_all);

% Unit shanks (already 1-index)
assert(isfield(spikes,"shankID") && ~isempty(spikes.shankID), 'Sorted spikes missing spikes.shankID.');
unitShank = double(spikes.shankID(:));

% Unit IDs (for display)
if isfield(spikes,"UID") && ~isempty(spikes.UID)
    unitIDs = double(spikes.UID(:));
elseif isfield(spikes,"cluID") && ~isempty(spikes.cluID)
    unitIDs = double(spikes.cluID(:));
else
    unitIDs = (1:nUnitsTotal).';
end

%% ============================================================
%  Prepare per-shank active units in the window
% ============================================================

activeUnitIdxByShank = cell(nShanks,1);
activeUnitIDsByShank = cell(nShanks,1);
nActiveByShank = zeros(nShanks,1);

for s = 1:nShanks
    unitIdx = find(unitShank == s);
    isActive = false(size(unitIdx));

    for k = 1:numel(unitIdx)
        u = unitIdx(k);
        t_u = double(tcell_all{u});
        if isempty(t_u), continue; end
        inWin = (t_u >= tStart_s) & (t_u <= tEnd_s);
        isActive(k) = any(inWin);
    end

    activeUnitIdx = unitIdx(isActive);
    activeUnitIdxByShank{s} = activeUnitIdx;
    activeUnitIDsByShank{s} = unitIDs(activeUnitIdx);
    nActiveByShank(s) = numel(activeUnitIdx);
end

%% ============================================================
%  (A) Combined raster: UNSORTED then SORTED for each shank (single axes)
% ============================================================

figName = sprintf('Combined raster | T0=%.1fs | first %d spikes | |Amp|>=%.1f uV', T0, Nspk, ampThresh_uV);
figure('Name',figName,'Color','w');
ax = axes(); hold(ax,'on');

% Collect the actual plotted raster points (both sorted and unsorted).
% Each row in the CSV corresponds to one plotted spike marker.
rasterTime = zeros(0,1);
rasterY = zeros(0,1);
rasterShank = zeros(0,1);
rasterType = strings(0,1);
rasterChannel = nan(0,1);       % original CSV channel (0-based)
rasterUnitID = nan(0,1);        % sorted UID, NaN for unsorted
rasterSpikeID = nan(0,1);      % 1..Nspk only for selected unsorted spikes
rasterHighlight = strings(0,1);
selectedIDByRow = nan(numel(time_s),1);
selectedIDByRow(idxSel) = (1:numel(idxSel)).';

y0 = 0;  % running y-offset (in rows)

for s = 1:nShanks
    % ---- SORTED: only ACTIVE units on this shank (BOTTOM of this shank block) ----
    activeUnitIdx = activeUnitIdxByShank{s};
    nU = numel(activeUnitIdx);

    if nU > 0
        % High-contrast categorical palette
        unitColors = lines(nU);

        for k = 1:nU
            u = activeUnitIdx(k);
            t_u = double(tcell_all{u});
            if isempty(t_u), continue; end
            t_u = t_u((t_u >= tStart_s) & (t_u <= tEnd_s));
            if isempty(t_u), continue; end

            y_u = y0 + k;   % sorted rows at bottom of this shank block
            plot(ax, t_u, y_u*ones(size(t_u)), '^', 'MarkerEdgeColor', unitColors(k,:), 'MarkerFaceColor', unitColors(k,:), 'MarkerSize', ms_sorted);
            nPts = numel(t_u);
            rasterTime = [rasterTime; t_u(:)];
            rasterY = [rasterY; repmat(y_u,nPts,1)];
            rasterShank = [rasterShank; repmat(s,nPts,1)];
            rasterType = [rasterType; repmat("Sorted",nPts,1)];
            rasterChannel = [rasterChannel; nan(nPts,1)];
            rasterUnitID = [rasterUnitID; repmat(unitIDs(u),nPts,1)];
            rasterSpikeID = [rasterSpikeID; nan(nPts,1)];
            rasterHighlight = [rasterHighlight; repmat("Unit color",nPts,1)];
        end
    end

    % Separator between sorted and unsorted within this shank
    % Place it midway between the top sorted row (y0+nU) and the first unsorted row.
    sep1 = y0 + nU + 0.5*unsortedRowSpacing;
    yline(ax, sep1, '--', 'Color', [0.65 0.65 0.65]);

    % ---- UNSORTED: keep suprathreshold spikes in window AND on this shank (ABOVE sorted) ----
    keep = (time_s >= tStart_s) & (time_s <= tEnd_s) & (shank_csv == s) & (abs(amp_uV) >= ampThresh_uV);

    t_plot = time_s(keep);
    ch_plot = chan(keep);

    % Global indices in the FULL (time-sorted) CSV arrays for these UNSORTED points
    idx_keep_all = find(keep);

    % local channel within shank: 1..16 (assumes 0..63 global indexing)
    % Apply user-controlled vertical spacing only to UNSORTED rows.
    y_uns = y0 + nU + (mod(ch_plot, rowsPerShank) + 1) * unsortedRowSpacing;

    % Split into highlighted vs default points (by GLOBAL index membership)
    % Split into highlighted vs default points (by GLOBAL index membership)
    isHL1 = false(size(idx_keep_all));
    isHL2 = false(size(idx_keep_all));

    % highlightSpikeIdx1 are spike IDs within the selected window (1..Nspk)
    if ~isempty(highlightSpikeIdx1)
        hl1 = highlightSpikeIdx1(:);
        hl1 = hl1(hl1>=1 & hl1<=numel(idxSel));
        hl1_global = idxSel(hl1);                 % GLOBAL indices into FULL CSV arrays
        isHL1 = ismember(idx_keep_all, hl1_global);
    end

    % highlightSpikeIdx2 are spike IDs within the selected window (1..Nspk)
    if ~isempty(highlightSpikeIdx2)
        hl2 = highlightSpikeIdx2(:);
        hl2 = hl2(hl2>=1 & hl2<=numel(idxSel));
        hl2_global = idxSel(hl2);                 % GLOBAL indices into FULL CSV arrays
        isHL2 = ismember(idx_keep_all, hl2_global);
    end

    % If overlap: let list 2 take precedence (so spikes in both use color2)
    isHL1 = isHL1 & ~isHL2;

    if ~isempty(t_plot)
        % Default (black): not in either highlight list
        isDefault = ~(isHL1 | isHL2);
        if any(isDefault)
            plot(ax, t_plot(isDefault), y_uns(isDefault), '.', 'Color', [0 0 0], 'MarkerSize', ms_unsorted);
        end

        % Highlight set 1
        if any(isHL1)
            plot(ax, t_plot(isHL1), y_uns(isHL1), '.', 'Color', highlightColor1, 'MarkerSize', ms_unsorted);
        end

        % Highlight set 2
        if any(isHL2)
            plot(ax, t_plot(isHL2), y_uns(isHL2), '.', 'Color', highlightColor2, 'MarkerSize', ms_unsorted);
        end
    end


    % Record the exact unsorted markers, including plotted Y coordinate
    % and their highlight grouping (set 2 takes precedence).
    nPts = numel(t_plot);
    hlLabel = repmat("Default",nPts,1);
    hlLabel(isHL1) = "Highlight 1";
    hlLabel(isHL2) = "Highlight 2";
    rasterTime = [rasterTime; t_plot(:)];
    rasterY = [rasterY; y_uns(:)];
    rasterShank = [rasterShank; repmat(s,nPts,1)];
    rasterType = [rasterType; repmat("Unsorted",nPts,1)];
    rasterChannel = [rasterChannel; ch_plot(:)];
    rasterUnitID = [rasterUnitID; nan(nPts,1)];
    rasterSpikeID = [rasterSpikeID; selectedIDByRow(idx_keep_all)];
    rasterHighlight = [rasterHighlight; hlLabel];

    % Separator after the unsorted block for this shank
    unsH = rowsPerShank * unsortedRowSpacing;
    sep2 = y0 + nU + unsH + 0.5*unsortedRowSpacing;
    yline(ax, sep2, '--', 'Color', [0.65 0.65 0.65]);

    % Shank label (optional)
    yMid = y0 + (nU + unsH)/2;
    text(ax, tStart_s, yMid, sprintf('  Shank %d', s), 'VerticalAlignment', ...
        'middle', 'FontWeight','bold');

    % Advance offset: nU sorted rows + (16 * spacing) unsorted rows + gap
    y0 = y0 + nU + unsH + gapRows;
end

xlim(ax, [tStart_s tEnd_s]);
ylim(ax, [0 y0]);
xlabel(ax,'Time (s)');
ylabel(ax,'Units (sorted) + Channels (unsorted), stacked by shank');
set(ax,'YDir','normal');   % chan0 at bottom (physical)
box(ax,'off');

title(ax, sprintf('Combined raster | Window %.3f–%.3f s | Sorted (colored) below Unsorted (BLACK)', tStart_s, tEnd_s));

if doSave
    save_label_variants(gcf, outDir4A, '4A_combined_raster');
end

% Export the points underlying Figure 4A (not an image of the plot).
if doExportCsv
    rasterTable = table(rasterTime, rasterY, rasterShank, rasterType, ...
        rasterChannel, rasterUnitID, rasterSpikeID, rasterHighlight, ...
        'VariableNames', {'Time_s','RasterY','Shank','SpikeType', ...
        'Channel_0based','SortedUnitID','SelectedSpikeID','HighlightGroup'});
    rasterCsv = fullfile(outDir4A, '4A_combined_raster.csv');
    writetable(rasterTable, rasterCsv);
    fprintf('Saved raster source data: %s (%d markers)\n', rasterCsv, height(rasterTable));
end

%% ============================================================
%  (B) Feature traces (RAW) for the selected Nspk spikes
% ============================================================

% Pick feature columns if present
featCols = {'Amplitude (uV)','Peak A (uV)','ISI (ms)','Trough to Peak (ms)'};
have = ismember(featCols, T.Properties.VariableNames);

hTrace = figure('Name','Feature traces (raw)','Color','w');

% Cache selected feature vectors for optional histograms
featSel = cell(size(featCols));

for i = 1:numel(featCols)
    ax_i = subplot(numel(featCols),1,i); hold on;

    if have(i)
        % ---- RAW values only ----
        v = double(T.(featCols{i}));
        vSel = v(idxSel);   % NO normalization
        featSel{i} = vSel;  % Store selected spikes for histogram plotting

        spikeID = (1:numel(vSel)).';

        % ---- Scatter only (black) ----
        scatter(spikeID, vSel, 10, 'k', 'filled');

        % ---- Highlight set 1 ----
        if ~isempty(highlightSpikeIdx1)
            hl1 = highlightSpikeIdx1(:);
            hl1 = hl1(hl1>=1 & hl1<=numel(vSel));
            if ~isempty(hl1)
                scatter(spikeID(hl1), vSel(hl1), 20, highlightColor1, 'filled');
            end
        end

        % ---- Highlight set 2 ----
        if ~isempty(highlightSpikeIdx2)
            hl2 = highlightSpikeIdx2(:);
            hl2 = hl2(hl2>=1 & hl2<=numel(vSel));
            if ~isempty(hl2)
                scatter(spikeID(hl2), vSel(hl2), 20, highlightColor2, 'filled');
            end
        end

        % ---- Axis formatting ----
        vmin = min(vSel);
        vmax = max(vSel);
        if vmin == vmax
            pad = 1;
        else
            pad = 0.05*(vmax - vmin);
        end
        ylim([vmin-pad, vmax+pad]);
        xlim([1 numel(vSel)]);
        ylabel(featCols{i}, 'Interpreter','none');

        % ---- Only bottom subplot shows X label ----
        if i == numel(featCols)
            xlabel('Spike ID');
        else
            set(ax_i,'XTickLabel',[]);
        end

        box off;
        set(ax_i,'TickDir','out');

    else
        text(0.1,0.5, sprintf('Missing column: %s', featCols{i}), 'Units','normalized');
        axis off;
    end
end


% Export Figure 4E's four raw feature vectors into ONE CSV file,
% with one row per selected spike and the same 1..Nspk x-axis IDs.
if doExportCsv
    featureTable = table((1:numel(idxSel)).', time_s(idxSel), ...
        'VariableNames', {'SpikeID','Time_s'});
    excelFeatureNames = {'Amplitude_uV','PeakA_uV','ISI_ms','TroughToPeak_ms'};
    for i = 1:numel(featCols)
        if have(i)
            featureTable.(excelFeatureNames{i}) = featSel{i}(:);
        else
            featureTable.(excelFeatureNames{i}) = nan(numel(idxSel),1);
        end
    end
    featureHL = repmat("Default",numel(idxSel),1);
    h1 = highlightSpikeIdx1(:);
    h1 = h1(h1>=1 & h1<=numel(idxSel));
    h2 = highlightSpikeIdx2(:);
    h2 = h2(h2>=1 & h2<=numel(idxSel));
    featureHL(h1) = "Highlight 1";
    featureHL(h2) = "Highlight 2";  % highlight 2 drawn last
    featureTable.HighlightGroup = featureHL;
    featureCsv = fullfile(outDir4E, '4E_feature_traces.csv');
    writetable(featureTable, featureCsv);
    fprintf('Saved feature source data: %s (%d spikes)\n', featureCsv, height(featureTable));
end

%% ============================================================
%  (C) Feature histograms for the SAME selected Nspk spikes
%      4 separate figures (one per feature)
%      Darker gray bins + black edge
% ============================================================

if doFeatureHists
    % Histogram styling
    hist_face = [0.35 0.35 0.35];     % dark gray fill
    hist_edge = 'k';                  % black edge
    hist_edge_alpha = 0.25;           % soft edge
    hist_linewidth = 0.75;

    for i = 1:numel(featCols)
        if ~(have(i) && ~isempty(featSel{i}))
            fprintf('[Info] Histogram skipped (missing/empty): %s\n', featCols{i});
            continue;
        end

        data = double(featSel{i});
        data = data(isfinite(data));   % keep raw values; just drop NaN/Inf
        if isempty(data)
            fprintf('[Info] Histogram skipped (no finite values): %s\n', featCols{i});
            continue;
        end

        hHist = figure('Name',sprintf('Histogram: %s', featCols{i}), 'Color','w');
        axH = axes(hHist); hold(axH,'on');

        h = histogram(axH, data, histNBins, ...
            'FaceColor', hist_face, ...
            'EdgeColor', hist_edge, ...
            'LineWidth', hist_linewidth);

        ylabel(axH,'Count');
        xlabel(axH, featCols{i}, 'Interpreter','none');
        title(axH, sprintf('%s (n=%d)', featCols{i}, numel(data)), 'Interpreter','none');
        box(axH,'off');
        set(axH,'TickDir','out');

        if doSave
            safeName = regexprep(featCols{i}, '[^a-zA-Z0-9]+', '_');
            save_label_variants(hHist, outDir4E, sprintf('4E_feature_hist_%s', safeName));
        end
    end
end

if doSave
    save_label_variants(hTrace, outDir4E, '4E_feature_traces');
end

%% ============================================================
%  (D) Single-unit waveforms + print spike times + channel (by CluID/UID)
% ============================================================
%
% This section prompts for a unit ID (CluID / UID as shown in the sorted raster),
% then prints:
%   - unit ID
%   - channel (best-effort; depends on what fields exist in spikes struct)
%   - spike timestamps (seconds) WITHIN the selected window
%
% It also attempts to plot stored waveforms if they are present in the MAT file.

try
    promptStr = sprintf('\nEnter a UID (not cluID) to inspect (empty to skip): ');
    cluIn = input(promptStr);
catch
    cluIn = [];
end

if ~isempty(cluIn)
    % Map the requested ID -> unit index in spikes
    uReq = find(unitIDs == double(cluIn), 1, 'first');
    if isempty(uReq)
        fprintf('No unit found with ID = %g (check IDs in sorted raster).\n', cluIn);
    else
        % Unit channel (best-effort)
        chan_u = NaN;
        if isfield(spikes,'maxWaveformCh') && numel(spikes.maxWaveformCh) >= uReq
            chan_u = double(spikes.maxWaveformCh(uReq));
        elseif isfield(spikes,'maxWaveformCh1') && numel(spikes.maxWaveformCh1) >= uReq
            chan_u = double(spikes.maxWaveformCh1(uReq));
        elseif isfield(spikes,'maxChannel') && numel(spikes.maxChannel) >= uReq
            chan_u = double(spikes.maxChannel(uReq));
        elseif isfield(spikes,'maxwaveformCh') && numel(spikes.maxwaveformCh) >= uReq
            chan_u = double(spikes.maxwaveformCh(uReq));
        end

        % Spike times in window
        t_u_all = double(tcell_all{uReq});
        t_u_win = t_u_all((t_u_all >= tStart_s) & (t_u_all <= tEnd_s));

        fprintf('\n===== Unit inspection =====\n');
        fprintf('Unit ID (CluID/UID): %g\n', unitIDs(uReq));
        if isfinite(chan_u)
            fprintf('Channel (unit): %g\n', chan_u);
        else
            fprintf('Channel (unit): [not found in spikes struct]\n');
        end
        fprintf('Num spikes in window: %d\n', numel(t_u_win));

        if ~isempty(t_u_win)
            % Print as a small table in the command window
            fprintf('  Spike#\tTime_s\t\tChannel\n');
            for ii = 1:numel(t_u_win)
                if isfinite(chan_u)
                    fprintf('  %4d\t\t%.6f\t%g\n', ii, t_u_win(ii), chan_u);
                else
                    fprintf('  %4d\t\t%.6f\t\t\n', ii, t_u_win(ii));
                end
            end
        end

        % Attempt to plot waveforms (field names vary by pipeline)
        wf = [];
        if isfield(spikes,'waveforms') && ~isempty(spikes.waveforms)
            wf = spikes.waveforms{uReq};
        elseif isfield(spikes,'rawWaveforms') && ~isempty(spikes.rawWaveforms)
            wf = spikes.rawWaveforms{uReq};
        elseif isfield(spikes,'filtWaveforms') && ~isempty(spikes.filtWaveforms)
            wf = spikes.filtWaveforms{uReq};
        elseif isfield(spikes,'filtWaveform') && ~isempty(spikes.filtWaveform)
            wf = spikes.filtWaveform{uReq};
        end

        if ~isempty(wf)
            figure('Name',sprintf('Unit waveforms: ID %g', unitIDs(uReq)),'Color','w');
            plot(wf','k-');  % each row = waveform
            title(sprintf('Unit waveforms | ID %g | n=%d', unitIDs(uReq), size(wf,1)));
            xlabel('Sample');
            ylabel('uV (if scaled)');
            box off;
        else
            fprintf('[Info] Waveform plotting skipped: no waveform field found in spikes struct.\n');
        end
    end
end
