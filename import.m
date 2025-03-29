function import()
    addpath(pwd());
    addpath('./Lie_group')
    addpath('./Stiefel')
        
    % Ask user if the path should be saved or not
    fprintf('VariationalOpt import success!\n');
    pause(0.5)
    fprintf('Save path for future Matlab sessions? [Y/N]\n');
    response = input("", 's');
    if strcmpi(response, 'Y')
        failed = savepath();
        if ~failed
            fprintf('Path saved.\n');
        else
            fprintf(['Failed\n']);
        end
    else
        fprintf('Path not saved.\n');
    end
end