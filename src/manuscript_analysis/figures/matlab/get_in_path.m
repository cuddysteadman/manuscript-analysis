function in = get_in_path(varargin)
%GET_IN_PATH Return <project root>/input/<varargin{:}>.
%   Counterpart of get_out_path.m for reading data. Nothing is created.
%   The project root is the nearest parent folder containing pyproject.toml.
%
%   in = get_in_path('batch_units_depth_struct.mat')   ->   <project root>\input\batch_units_depth_struct.mat

in = fullfile(find_project_root(), 'input', varargin{:});
end
