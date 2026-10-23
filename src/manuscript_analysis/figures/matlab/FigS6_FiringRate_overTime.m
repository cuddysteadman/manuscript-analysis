%% ============================================================
%  FiringRate_overTime.m  (SCRIPT)
%
%  Comm. BIo.: Fig. S6:
%    1. Plot firing rate for adjacent electrodes during insertion.
%    2. Use identical Y-axis limits for all channels.
%    3. Show motor movements as vertical red lines.
%    4. Quantify two hypotheses:
%
%       MODEL 1 - Tissue dragging
%       Same electrode should retain similar activity after movement:
%           electrode e @ step s
%                   vs.
%           electrode e @ step s+1
%
%       MODEL 2 - Smooth insertion
%       With electrode pitch = motor step = 20 um, the next shallower
%       electrode should occupy approximately the previous location:
%           electrode e @ step s
%                   vs.
%           electrode e+1 @ step s+1
%
%  Mean firing rate for each electrode/step is calculated using
%  the SECOND HALF of each insertion step.
%
%  Input is read from input/s6_firing_rate.csv in the project. Outputs go to output/figures/S6A, S6B and S6C
%  in the project as PNG/SVG/PDF with and without labels plus CSV source data
%  (get_out_path.m, save_label_variants.m).
%
% ============================================================

clear; clc;
addpath(fileparts(mfilename('fullpath')));  % get_out_path.m, save_label_variants.m

%% ============================================================
%  USER PARAMETERS
% ============================================================

% IMPORTANT:
% Enter channels in PHYSICAL ORDER FROM PROBE TIP -> BASE.
%
% With this ordering:
%   channels(1) is closest to the probe tip and enters first.


channelsTipToBase = 49:55;

t0_s  = 170;
t1_s  = 245;

% Bin size for firing-rate traces
bin_s = 0.1;

fs      = 20000;
projRoot = fileparts(fileparts(get_out_path()));   % <project root>/output/figures -> <project root>
csvFile = fullfile(projRoot, 'input', 's6_firing_rate.csv');   % Sample + Channel of processed_features.csv, channels 49-55, 170-245 s

sampleColWanted = "Sample";
chanColWanted   = "Channel";


%% ---------------- MOTOR MOVEMENT TIMES ----------------

motorStart_s = 180;
motorStepInterval_s = 6.54;
nMotorSteps = 10;

motorMoveTimes_s = motorStart_s + (0:nMotorSteps-1)*motorStepInterval_s;

% End of final insertion step used for model analysis.
% Usually this can simply be the end of the plotted/usable period.
analysisEnd_s = motorMoveTimes_s(end) + motorStepInterval_s;


%% ---------------- FIGURE SETTINGS ----------------

% Identical firing-rate Y-axis for every channel
yLimRate = [0 120];

% Requested ticks
yTicksRate = [0 50 100];

% Width of rate traces
traceLineWidth = 1.5;

% Motor movement line width
movementLineWidth = 1.2;


%% ---------------- MODEL ANALYSIS SETTINGS ----------------

% Exclude pairs containing zero firing rate.
%
% This follows the suggested analysis where electrodes outside
% the active zone should not dominate the comparison.
excludeZeroPairs = true;

% Optional minimum firing rate instead of exactly zero.
% Leave at 0 to remove only zero-rate measurements.
minRateForPair = 0;


%% ============================================================
%  READ HEADER MANUALLY
% ============================================================

fid = fopen(csvFile,'r');

if fid == -1
    error('Could not open %s', csvFile);
end

headerLine = fgetl(fid);
fclose(fid);

headers = string(strsplit(headerLine, ','));

iSample = find(headers == sampleColWanted,1);
iChan   = find(headers == chanColWanted,1);

if isempty(iSample) || isempty(iChan)
    error('Could not find Sample or Channel in CSV header.');
end


%% ============================================================
%  READ DATA
% ============================================================

T = readtable(csvFile, ...
    'Delimiter', ',', ...
    'HeaderLines', 1, ...
    'ReadVariableNames', false);

samples = double(T{:,iSample});
chans   = double(T{:,iChan});


%% ============================================================
%  CONVERT SAMPLE NUMBER -> TIME
% ============================================================

time_s = samples ./ fs;

maskTime = ...
    (time_s >= t0_s) & ...
    (time_s <= t1_s);

time_s = time_s(maskTime);
chans  = chans(maskTime);


%% ============================================================
%  FIRING-RATE BINS
% ============================================================

binEdges = t0_s:bin_s:t1_s;

if binEdges(end) < t1_s
    binEdges(end+1) = t1_s;
end

binCenters = ...
    binEdges(1:end-1) + diff(binEdges)/2;


%% ============================================================
%  CALCULATE FIRING RATE FOR ALL CHANNELS
% ============================================================

nCh = numel(channelsTipToBase);

allRates = zeros(nCh,numel(binCenters));

for i = 1:nCh

    ch = channelsTipToBase(i);

    idx = chans == ch;
    ch_times = time_s(idx);

    if isempty(ch_times)

        rates = zeros(1,numel(binCenters));

    else

        counts = histcounts(ch_times,binEdges);
        rates  = counts ./ bin_s;

    end

    allRates(i,:) = rates;

end


%% ============================================================
%  FIGURE S6
% ============================================================
%
% Plot physical order with probe-tip electrode first.
% To put the tip electrode at the TOP of the figure, subplot
% index follows channelsTipToBase directly.
%

%% ============================================================
%  FIGURE S6
%  Plot channels in reverse order:
%  highest channel at top, Channel 49 at bottom
% ============================================================

plotOrder = fliplr(channelsTipToBase);

figS6A = figure('Color','w');

for i = 1:nCh

    ch = plotOrder(i);

    % Find corresponding row in allRates
    rateIdx = find(channelsTipToBase == ch);

    ax = subplot(nCh,1,i);

    plot(binCenters, ...
         allRates(rateIdx,:), ...
         'k', ...
         'LineWidth',traceLineWidth);

    hold on;

    % Motor movements
    for m = 1:numel(motorMoveTimes_s)
        xline(motorMoveTimes_s(m), ...
            'r-', ...
            'LineWidth',movementLineWidth, ...
            'HandleVisibility','off');
    end

    xlim([t0_s t1_s]);
    ylim(yLimRate);
    yticks(yTicksRate);

    box off;

    set(gca, ...
        'FontSize',12, ...
        'TickDir','out');

    title(sprintf('Channel %d',ch));
    ylabel('Firing rate [spk/s]');

    if i < nCh
        set(gca,'XTickLabel',[]);
    else
        xlabel('Time [s]');
    end
end

sgtitle(sprintf( ...
    'Firing rate during probe insertion (bin = %.2f s)', ...
    bin_s));

% Source data for Figure S6A: one row per channel and time bin.
outDirS6A = get_out_path('S6A');
rateTable = table( ...
    repelem(channelsTipToBase(:), numel(binCenters)), ...
    repmat(binCenters(:), nCh, 1), ...
    reshape(allRates.', [], 1), ...
    'VariableNames', {'Channel', 'Time_s', 'FiringRate_spk_per_s'});
writetable(rateTable, fullfile(outDirS6A, 'S6A_firing_rate_over_time.csv'));
save_label_variants(figS6A, outDirS6A, 'S6A_firing_rate_over_time');

%% ============================================================
%  MODEL ANALYSIS
%
%  Requires motor movement timestamps.
% ============================================================

if numel(motorMoveTimes_s) < 2

    warning(['Model analysis skipped because motorMoveTimes_s ', ...
             'contains fewer than two movement times.']);

else

    %% --------------------------------------------------------
    % Define insertion steps
    %
    % Step s:
    %   motorMoveTimes_s(s) -> motorMoveTimes_s(s+1)
    %
    % Final step:
    %   last motorMoveTimes_s -> analysisEnd_s
    % ---------------------------------------------------------

    stepStart = motorMoveTimes_s(:)';

    stepEnd = [ ...
        motorMoveTimes_s(2:end), ...
        analysisEnd_s ...
        ];

    nSteps = numel(stepStart);

    if analysisEnd_s <= stepStart(end)

        error(['analysisEnd_s must occur after the final ', ...
               'motor movement.']);

    end


    %% --------------------------------------------------------
    % Calculate mean firing rate during SECOND HALF of each step
    %
    % rateSecondHalf(e,s)
    %
    % e = electrode index, ordered tip -> base
    % s = insertion step
    % ---------------------------------------------------------

    rateSecondHalf = nan(nCh,nSteps);

    fprintf('\n');
    fprintf('====================================================\n');
    fprintf('MEAN FIRING RATE: SECOND HALF OF EACH STEP\n');
    fprintf('====================================================\n');


    for s = 1:nSteps

        stepDuration = stepEnd(s) - stepStart(s);

        halfStart = ...
            stepStart(s) + stepDuration/2;

        halfEnd = stepEnd(s);

        fprintf( ...
            'Step %d: %.3f to %.3f s; analysis window %.3f to %.3f s\n', ...
            s, ...
            stepStart(s), ...
            stepEnd(s), ...
            halfStart, ...
            halfEnd);


        for e = 1:nCh

            ch = channelsTipToBase(e);

            idx = ...
                chans == ch & ...
                time_s >= halfStart & ...
                time_s < halfEnd;

            spikeCount = sum(idx);

            duration_s = ...
                halfEnd - halfStart;

            rateSecondHalf(e,s) = ...
                spikeCount / duration_s;

        end

    end


    %% ========================================================
    %  DISPLAY STEP-RATE MATRIX
    % ========================================================

    fprintf('\nMean firing rates [spk/s]\n');
    fprintf('Rows = electrodes, tip -> base\n');
    fprintf('Columns = insertion steps\n\n');

    disp(rateSecondHalf);


    %% ========================================================
    %  MODEL 1
    %
    %  TISSUE DRAGGING MODEL
    %
    %  Same electrode across consecutive steps:
    %
    %       e1,s1  -> e1,s2
    %       e1,s2  -> e1,s3
    %       e2,s1  -> e2,s2
    %       ...
    %
    %  Number before exclusion:
    %
    %       K * (N-1)
    %
    % ========================================================

    x_model1 = [];
    y_model1 = [];

    electrode_model1 = [];
    step_model1 = [];


    for e = 1:nCh

        for s = 1:(nSteps-1)

            x_model1(end+1,1) = ...
                rateSecondHalf(e,s);

            y_model1(end+1,1) = ...
                rateSecondHalf(e,s+1);

            electrode_model1(end+1,1) = e;
            step_model1(end+1,1) = s;

        end

    end


    %% Remove invalid/zero pairs

    valid1 = ...
        isfinite(x_model1) & ...
        isfinite(y_model1);

    if excludeZeroPairs

        valid1 = ...
            valid1 & ...
            x_model1 > minRateForPair & ...
            y_model1 > minRateForPair;

    end

    x_model1 = x_model1(valid1);
    y_model1 = y_model1(valid1);

    electrode_model1 = electrode_model1(valid1);
    step_model1 = step_model1(valid1);


    %% ========================================================
    %  MODEL 2
    %
    %  SMOOTH INSERTION MODEL
    %
    %  Because:
    %
    %      electrode spacing = 20 um
    %      insertion step     = 20 um
    %
    %  electrode e+1 after the movement should occupy
    %  approximately the position previously occupied by e.
    %
    %       e1,s1 -> e2,s2
    %       e1,s2 -> e2,s3
    %       e2,s1 -> e3,s2
    %       ...
    %
    %  Number before exclusion:
    %
    %       (K-1) * (N-1)
    %
    % ========================================================

    x_model2 = [];
    y_model2 = [];

    electrode_model2 = [];
    step_model2 = [];


    for e = 1:(nCh-1)

        for s = 1:(nSteps-1)

            x_model2(end+1,1) = ...
                rateSecondHalf(e,s);

            y_model2(end+1,1) = ...
                rateSecondHalf(e+1,s+1);

            electrode_model2(end+1,1) = e;
            step_model2(end+1,1) = s;

        end

    end


    %% Remove invalid/zero pairs

    valid2 = ...
        isfinite(x_model2) & ...
        isfinite(y_model2);

    if excludeZeroPairs

        valid2 = ...
            valid2 & ...
            x_model2 > minRateForPair & ...
            y_model2 > minRateForPair;

    end

    x_model2 = x_model2(valid2);
    y_model2 = y_model2(valid2);

    electrode_model2 = electrode_model2(valid2);
    step_model2 = step_model2(valid2);


    %% ========================================================
    %  FIT MODEL 1
    % ========================================================

    stats1 = calculateIdentityStats( ...
        x_model1, ...
        y_model1);


    %% ========================================================
    %  FIT MODEL 2
    % ========================================================

    stats2 = calculateIdentityStats( ...
        x_model2, ...
        y_model2);


    %% ========================================================
    %  SCATTER PLOT - MODEL 1
    % ========================================================

    figS6B = figure('Color','w');

    scatter( ...
        x_model1, ...
        y_model1, ...
        45, ...
        'filled');

    hold on;


    maxVal1 = max([ ...
        x_model1; ...
        y_model1 ...
        ]);

    if isempty(maxVal1) || maxVal1 == 0
        maxVal1 = 1;
    end


    %% Identity line

    plot( ...
        [0 maxVal1], ...
        [0 maxVal1], ...
        'k--', ...
        'LineWidth',1.5);


    %% Regression line

    xFit = linspace(0,maxVal1,200);

    yFit = ...
        stats1.slope*xFit + ...
        stats1.intercept;

    plot( ...
        xFit, ...
        yFit, ...
        'k-', ...
        'LineWidth',2);


    axis square;

    xlim([0 maxVal1]);
    ylim([0 maxVal1]);

    xlabel( ...
        'Firing rate at step s [spk/s]');

    ylabel( ...
        'Same-electrode firing rate at step s+1 [spk/s]');

    title({ ...
        'Model 1: tissue dragging', ...
        sprintf('y = %.3fx %+.3f, R^2 = %.3f', ...
        stats1.slope, ...
        stats1.intercept, ...
        stats1.R2) ...
        });

    box off;
    set(gca,'FontSize',14,'TickDir','out');

    save_label_variants(figS6B, get_out_path('S6B'), 'S6B_model1_tissue_dragging');


    %% ========================================================
    %  SCATTER PLOT - MODEL 2
    % ========================================================

    figS6C = figure('Color','w');

    scatter( ...
        x_model2, ...
        y_model2, ...
        45, ...
        'filled');

    hold on;


    maxVal2 = max([ ...
        x_model2; ...
        y_model2 ...
        ]);

    if isempty(maxVal2) || maxVal2 == 0
        maxVal2 = 1;
    end


    %% Identity line

    plot( ...
        [0 maxVal2], ...
        [0 maxVal2], ...
        'k--', ...
        'LineWidth',1.5);


    %% Regression line

    xFit = linspace(0,maxVal2,200);

    yFit = ...
        stats2.slope*xFit + ...
        stats2.intercept;

    plot( ...
        xFit, ...
        yFit, ...
        'k-', ...
        'LineWidth',2);


    axis square;

    xlim([0 maxVal2]);
    ylim([0 maxVal2]);

    xlabel( ...
        'Firing rate at step s [spk/s]');

    ylabel( ...
        'Next-electrode firing rate at step s+1 [spk/s]');

    title({ ...
        'Model 2: smooth insertion', ...
        sprintf('y = %.3fx %+.3f, R^2 = %.3f', ...
        stats2.slope, ...
        stats2.intercept, ...
        stats2.R2) ...
        });

    box off;
    set(gca,'FontSize',14,'TickDir','out');

    save_label_variants(figS6C, get_out_path('S6C'), 'S6C_model2_smooth_insertion');


    %% ========================================================
    %  PRINT RESULTS
    % ========================================================

    fprintf('\n');
    fprintf('====================================================\n');
    fprintf('MODEL COMPARISON\n');
    fprintf('====================================================\n');


    fprintf('\nMODEL 1: SAME ELECTRODE / TISSUE DRAGGING\n');

    fprintf('Number of included pairs: %d\n', ...
        stats1.n);

    fprintf('Slope:      %.4f\n', ...
        stats1.slope);

    fprintf('Intercept:  %.4f spk/s\n', ...
        stats1.intercept);

    fprintf('R^2:        %.4f\n', ...
        stats1.R2);

    fprintf('Identity MAE:  %.4f spk/s\n', ...
        stats1.identityMAE);

    fprintf('Identity RMSE: %.4f spk/s\n', ...
        stats1.identityRMSE);


    fprintf('\nMODEL 2: SHIFTED ELECTRODE / SMOOTH INSERTION\n');

    fprintf('Number of included pairs: %d\n', ...
        stats2.n);

    fprintf('Slope:      %.4f\n', ...
        stats2.slope);

    fprintf('Intercept:  %.4f spk/s\n', ...
        stats2.intercept);

    fprintf('R^2:        %.4f\n', ...
        stats2.R2);

    fprintf('Identity MAE:  %.4f spk/s\n', ...
        stats2.identityMAE);

    fprintf('Identity RMSE: %.4f spk/s\n', ...
        stats2.identityRMSE);


    %% ========================================================
    %  SIMPLE INTERPRETATION
    % ========================================================

    fprintf('\n');
    fprintf('----------------------------------------------------\n');
    fprintf('Distance from the identity relationship y = x\n');
    fprintf('----------------------------------------------------\n');

    if stats1.identityMAE < stats2.identityMAE

        fprintf(['Model 1 has the smaller identity MAE ', ...
                 '(same-electrode pairing is closer to y=x).\n']);

    elseif stats2.identityMAE < stats1.identityMAE

        fprintf(['Model 2 has the smaller identity MAE ', ...
                 '(spatially shifted pairing is closer to y=x).\n']);

    else

        fprintf('The two models have identical identity MAE.\n');

    end


    %% ========================================================
    %  OPTIONAL TABLES OF INDIVIDUAL PAIRS
    % ========================================================

    model1Table = table( ...
        electrode_model1, ...
        step_model1, ...
        x_model1, ...
        y_model1, ...
        'VariableNames', { ...
        'Electrode', ...
        'Step', ...
        'Rate_step_s', ...
        'Rate_sameElectrode_step_sPlus1'});


    model2Table = table( ...
        electrode_model2, ...
        step_model2, ...
        x_model2, ...
        y_model2, ...
        'VariableNames', { ...
        'Electrode_at_step_s', ...
        'Step', ...
        'Rate_step_s', ...
        'Rate_nextElectrode_step_sPlus1'});


    fprintf('\nMODEL 1 PAIRS:\n');
    disp(model1Table);

    fprintf('\nMODEL 2 PAIRS:\n');
    disp(model2Table);

    writetable(model1Table, fullfile(get_out_path('S6B'), 'S6B_model1_tissue_dragging.csv'));
    writetable(model2Table, fullfile(get_out_path('S6C'), 'S6C_model2_smooth_insertion.csv'));

end


%% ============================================================
%  LOCAL FUNCTION
% ============================================================

function stats = calculateIdentityStats(x,y)

    x = x(:);
    y = y(:);

    valid = ...
        isfinite(x) & ...
        isfinite(y);

    x = x(valid);
    y = y(valid);

    stats.n = numel(x);

    if numel(x) < 2

        stats.slope = NaN;
        stats.intercept = NaN;
        stats.R2 = NaN;
        stats.identityMAE = NaN;
        stats.identityRMSE = NaN;

        return;

    end


    %% Ordinary least-squares fit:
    %
    % y = slope*x + intercept

    p = polyfit(x,y,1);

    stats.slope = p(1);
    stats.intercept = p(2);


    %% R-squared

    yPred = polyval(p,x);

    SSres = sum((y-yPred).^2);
    SStot = sum((y-mean(y)).^2);

    if SStot > 0
        stats.R2 = 1 - SSres/SStot;
    else
        stats.R2 = NaN;
    end


    %% Direct deviation from identity line y = x

    identityError = y-x;

    stats.identityMAE = ...
        mean(abs(identityError));

    stats.identityRMSE = ...
        sqrt(mean(identityError.^2));

end