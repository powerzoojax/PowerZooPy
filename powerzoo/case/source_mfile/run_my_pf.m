function run_my_pf(casename, output_dir)
% RUN_MY_PF  Run MATPOWER AC power flow and save results for Python testing.
%
% Usage:
%   run_my_pf                        % runs case33bw, saves to tests/ref_data/
%   run_my_pf('case33bw')
%   run_my_pf('case33bw', '/path/to/output')
%
% Requires: MATPOWER (https://matpower.org) in the MATLAB path.
%
% Output files in output_dir:
%   <casename>_bus.csv    : bus_i, Vm (p.u.), Va (deg)
%   <casename>_branch.csv : fbus, tbus, status, Pf_MW, Qf_MVAr, Pt_MW, Qt_MVAr
%
% The CSV files are loaded by tests/test_case33bw_pf.py for validation.

if nargin < 1, casename = 'case33bw'; end
if nargin < 2
    script_dir = fileparts(mfilename('fullpath'));
    output_dir = fullfile(script_dir, '..', '..', '..', 'tests', 'ref_data');
end

% Locate case file (same directory as this script)
case_file = fullfile(fileparts(mfilename('fullpath')), casename);
if ~exist([case_file '.m'], 'file')
    error('Case file not found: %s.m', case_file);
end

fprintf('Running MATPOWER power flow: %s\n', casename);
mpopt = mpoption('verbose', 0, 'out.all', 0);
mpc = runpf(case_file, mpopt);

if ~mpc.success
    error('Power flow did not converge for %s', casename);
end

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ---- Bus results ----
% MATPOWER bus columns: BUS_I=1, VM=8, VA=9
bus_data = mpc.bus(:, [1, 8, 9]);
bus_file = fullfile(output_dir, [casename '_bus.csv']);
fid = fopen(bus_file, 'w');
fprintf(fid, 'bus_i,Vm,Va\n');
for i = 1:size(bus_data, 1)
    fprintf(fid, '%d,%.10f,%.10f\n', ...
        bus_data(i,1), bus_data(i,2), bus_data(i,3));
end
fclose(fid);

% ---- Branch results ----
% MATPOWER branch columns: F_BUS=1, T_BUS=2, BR_STATUS=11, PF=14, QF=15, PT=16, QT=17
branch_data = mpc.branch(:, [1, 2, 11, 14, 15, 16, 17]);
branch_file = fullfile(output_dir, [casename '_branch.csv']);
fid = fopen(branch_file, 'w');
fprintf(fid, 'fbus,tbus,status,Pf_MW,Qf_MVAr,Pt_MW,Qt_MVAr\n');
for i = 1:size(branch_data, 1)
    fprintf(fid, '%d,%d,%d,%.10f,%.10f,%.10f,%.10f\n', ...
        branch_data(i,1), branch_data(i,2), branch_data(i,3), ...
        branch_data(i,4), branch_data(i,5), ...
        branch_data(i,6), branch_data(i,7));
end
fclose(fid);

% ---- Summary ----
p_loss_mw = sum(mpc.branch(:,14) + mpc.branch(:,16)) / mpc.baseMVA * mpc.baseMVA;
% Net losses on in-service lines only
in_service = mpc.branch(:,11) == 1;
p_loss_mw  = sum(mpc.branch(in_service,14) + mpc.branch(in_service,16));
q_loss_mvar = sum(mpc.branch(in_service,15) + mpc.branch(in_service,17));
v_min = min(mpc.bus(:,8));
v_max = max(mpc.bus(:,8));
[~, v_min_bus_idx] = min(mpc.bus(:,8));
v_min_bus_i = mpc.bus(v_min_bus_idx, 1);

fprintf('  Converged:  yes\n');
fprintf('  BaseMVA:    %.1f\n', mpc.baseMVA);
fprintf('  Buses:      %d\n', size(mpc.bus, 1));
fprintf('  Branches:   %d total, %d in-service\n', ...
    size(mpc.branch,1), sum(in_service));
fprintf('  P loss:     %.4f MW\n', p_loss_mw);
fprintf('  Q loss:     %.4f MVAr\n', q_loss_mvar);
fprintf('  V_min:      %.6f p.u. at bus %d\n', v_min, v_min_bus_i);
fprintf('  V_max:      %.6f p.u.\n', v_max);
fprintf('\nResults saved to: %s\n', output_dir);
fprintf('  %s\n', bus_file);
fprintf('  %s\n', branch_file);
