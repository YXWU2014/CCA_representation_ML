clc; clear all;

%%
comb_D_E      = nchoosek([{'Co'} {'V'} {'Mn'} {'Mo'} {'Cu'} {'Nb'} {'W'} {'Ti'} {'Al'} {'Si'},{'Ta'}], 2);

source_folder   = '/Users/ywu/InSync/2021_MPIE/2021-12_H Diffusion/Matlab Toolbox HEA/v6_A-B-C-D-E_Sputtering_master/';
dst_folder      = '/Users/ywu/InSync/2021_MPIE/2021-12_H Diffusion/Matlab Toolbox HEA/v6_A-B-C-D-E_Sputtering_ML_master/';


for comb_i = 1: length(comb_D_E)

    comb_A_B_C_D_E_temp = ['Fe_Cr_Ni_',comb_D_E{comb_i,1}, '_', comb_D_E{comb_i,2}];

    %% --- source directory
    source_subfolder = ['v6_', comb_A_B_C_D_E_temp, '_Sputtering'];

    source_file = [source_folder source_subfolder '/SSS_FCC_byCompo.xlsx'];

    if exist(source_file, 'file')
        disp([comb_A_B_C_D_E_temp ': File exists!']);
    else
        disp([comb_A_B_C_D_E_temp ': File does not exist.']);
    end


    %% --- destination directory
    dst_subfolder = 'v6_A-B-C-D-E_Sputtering_ML_before';
    dst_file_at      = [dst_folder dst_subfolder '/', '','v6_', comb_A_B_C_D_E_temp, '_SSS_FCC_byCompo.xlsx'];


    %% --- copy
    status = copyfile(source_file, dst_file_at); % copy the file to the destination directory

    if status == 1
        disp([comb_A_B_C_D_E_temp ': File copied successfully!']);
    else
        disp('Error copying file.');
    end


end







