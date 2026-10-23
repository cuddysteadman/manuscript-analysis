function root = find_project_root()
%FIND_PROJECT_ROOT Return the nearest parent folder containing pyproject.toml.

here = fileparts(mfilename('fullpath'));
folder = here;
while true
    if exist(fullfile(folder, 'pyproject.toml'), 'file')
        root = folder;
        return;
    end
    parent = fileparts(folder);
    if strcmp(parent, folder)
        error('find_project_root:noProjectRoot', ...
            'Could not find the manuscript-analysis project root (no pyproject.toml above %s).', here);
    end
    folder = parent;
end
end
