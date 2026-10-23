function paths = save_label_variants(fig, outDir, stem)
%SAVE_LABEL_VARIANTS Save a figure with and without its text labels.
%   MATLAB counterpart of manuscript_analysis.figures.export.save_label_variants.
%   Writes <stem>_with_labels.{png,svg,pdf}, then hides every title, axis
%   label, tick label, legend and annotation and writes
%   <stem>_without_labels.{png,svg,pdf}. The labels are restored afterwards.
%   Returns the saved file paths as a cell array.
%
%   paths = save_label_variants(gcf, get_out_path('4A'), '4A_combined_raster')

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

paths = save_formats(fig, outDir, [stem '_with_labels']);

hidden = hide_labels(fig);
restore = onCleanup(@() restore_labels(hidden));
paths = [paths, save_formats(fig, outDir, [stem '_without_labels'])];
clear restore;

for k = 1:numel(paths)
    fprintf('Saved %s\n', paths{k});
end
end


function paths = save_formats(fig, outDir, stem)
png = fullfile(outDir, [stem '.png']);
svg = fullfile(outDir, [stem '.svg']);
pdf = fullfile(outDir, [stem '.pdf']);
exportgraphics(fig, png, 'Resolution', png_resolution(fig), 'BackgroundColor', 'white');
print(fig, svg, '-dsvg');
exportgraphics(fig, pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
paths = {png, svg, pdf};
end


function dpi = png_resolution(fig)
% 600 dpi like the Python figures, lowered for very large figures so the
% longer side stays under 20000 pixels.
maxPixels = 20000;
oldUnits = fig.Units;
fig.Units = 'inches';
longest = max(fig.Position(3:4));
fig.Units = oldUnits;
dpi = min(600, floor(maxPixels / longest));
end


function hidden = hide_labels(fig)
% Each row of "hidden" is {object, property, original value}, in the order
% the values must be put back (tick labels before their mode).
hidden = cell(0, 3);

texts = findall(fig, 'Type', 'text');
for k = 1:numel(texts)
    hidden(end+1, :) = {texts(k), 'Visible', texts(k).Visible}; %#ok<AGROW>
    texts(k).Visible = 'off';
end

legends = findall(fig, 'Type', 'legend');
for k = 1:numel(legends)
    hidden(end+1, :) = {legends(k), 'Visible', legends(k).Visible}; %#ok<AGROW>
    legends(k).Visible = 'off';
end

axesList = findall(fig, 'Type', 'axes');
for k = 1:numel(axesList)
    ax = axesList(k);
    for dim = {'X', 'Y', 'Z'}
        labelProp = [dim{1} 'TickLabel'];
        modeProp = [dim{1} 'TickLabelMode'];
        if isprop(ax, labelProp)
            hidden(end+1, :) = {ax, labelProp, ax.(labelProp)}; %#ok<AGROW>
            hidden(end+1, :) = {ax, modeProp, ax.(modeProp)}; %#ok<AGROW>
            ax.(labelProp) = {};
        end
    end
end

colorbars = findall(fig, 'Type', 'colorbar');
for k = 1:numel(colorbars)
    cb = colorbars(k);
    hidden(end+1, :) = {cb, 'TickLabels', cb.TickLabels}; %#ok<AGROW>
    hidden(end+1, :) = {cb, 'TickLabelsMode', cb.TickLabelsMode}; %#ok<AGROW>
    cb.TickLabels = {};
    hidden(end+1, :) = {cb.Label, 'Visible', cb.Label.Visible}; %#ok<AGROW>
    cb.Label.Visible = 'off';
end
end


function restore_labels(hidden)
for k = 1:size(hidden, 1)
    try
        obj = hidden{k, 1};
        obj.(hidden{k, 2}) = hidden{k, 3};
    catch
        % The object was deleted while the labels were hidden.
    end
end
end
