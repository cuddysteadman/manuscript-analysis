function out = get_out_path(varargin)
%GET_OUT_PATH Return <project root>/output/figures/<varargin{:}>, creating it.
%   MATLAB counterpart of manuscript_analysis.figures.fig_utils.get_out_path.
%   The project root is the nearest parent folder containing pyproject.toml.
%
%   out = get_out_path('4A')   ->   <project root>\output\figures\4A

root = find_project_root();
out = fullfile(root, 'output', 'figures', varargin{:});
if ~exist(out, 'dir')
    mkdir(out);
end
end
