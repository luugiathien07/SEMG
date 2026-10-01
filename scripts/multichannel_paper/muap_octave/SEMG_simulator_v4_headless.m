% Simulation of multi-sensors surface EMG
%
% Syntax:
%	[MP, SD, DD, NbofActiveMU, t, FR, DisPattern, sepMuEMG] = ...
%       SEMG_simulator_v4(LibraryName, Exitation, InSimPrd, fs, CV, varA, DeltaE, MinFR, RctMax)
% 
% Input:
%	- LibraryName -> MUAP Library file name: 
%                       - 'MUAPlib_baseline_fat10.mat' : 10mm fat layer
%                       - 'MUAPlib_baseline_fat4.mat'  :  4mm fat layer
%	- Exitation -> Mean excitatory drive (in %MVC)
%                       - if Exitation is a scalar: Constant MVC
%                       - otherwise Exitation should be a vector of time
%                           of sampling frequency: fs
%	- InSimPrd	-> Simulation period (in sec) (ignored if Exitation is a
%                   vector)
%	- fs		-> Sampling frequency (in Hz)
%	- CV		-> delay between the channels (in m.s^(-1))
%                   original MU library Conduction Velocity is 4m/s
%	- varA      -> variation rate of amplitude modulation (in percent)
%	- DeltaE	-> interelectrode distance (in meter)
%	- MinFR     -> minimum Firing Rate (in Hz)
%	- RctMax    -> maximum Recruitment (in %MVC)
%
% Output:
%	- MP                -> Monopolar recording (matrix*)
%	- SD                -> Single-differential recording (matrix*)
%	- DD                -> Double-differential recording (matrix*)
%	- NbofActiveMU      -> Number of active MUs along the time
%	- t                 -> Simulation time array
%	- FR(t,k)           -> Firing rate for each MU "k" along the time "t"
%	- DisPattern(t,k)   -> Markers of discharge for each MU "k" along the 
%                           time "t"
%	- sepMuEMG{k}(ch,t) -> Separated MP signal : MUAP "k", channel "ch" at
%                           time "t"
%
% * The m'th row corresponds to the m'th recording channel
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Modified by F. Leclerc the 12th of July 2007:           %%
%%%   Add of the parameters                                 %%
%%%   Fs      -> asked sample frequency (in Hz)             %%
%%%   DeltaE  -> interelectrode distance (in meter)         %%
%%%   CV      -> delay between the channels (in m.s^(-1))   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Modified by J. Roussel the 21th of March 2012:          %%
%%%   3.0 Simulation with variable exitation                %%
%%%   3.0 Add of outputs:                                   %%
%%%         FR          -> Firing rate of MUAPs             %%
%%%                           along the time                %%
%%%         DisPattern  -> Firing patern matrix of          %%
%%%                           MUAPs along the time          %%
%%%   3.1 Add of random amplitude modulation                %%
%%%   3.2 Add of the sepMuEMG                               %%
%%%                                                         %%
%%% The 11th of january 2013:                               %%
%%%   4.0 Jitter standard-deviation is modeled              %% 
%%%       using Clamann's law:                              %%
%%%           std-dev   =   0.00091 x mean�   +   4.0       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Example: Simulate an sEMG signal with a ramp exitation
%   % Simulation Parameters
%   Emin = 10;                               % Minimum Exitation (%MVC)
%   Emax = 80;                               % Maximum Exitation (%MVC)
%   InSimPrd = 2;                            % Duration of Simulation
%   fs = 1024*4;                             % Sampling Frequency
%   
%   % Exitation (in %MVC): RAMP from Emin to Emax
%   Exitation = Emin+(0:InSimPrd*fs-1)*(Emax-Emin)/(InSimPrd*fs-1);
%   
%   % Other Simulator Inputs
%   LibraryName = 'MUAPlib_baseline_fat4';   % 4mm fat layer library                                     
%   CV = 4;                     % Mean Conduction Velocity fixed to 4m/s
%   varA = 0;                   % No random amplitude variation (0%)
%   DeltaE = 0.005;             % inter-electrode distance (in meter): 5mm
%   MinFR = 8;                  % Minimum Firing Rate: 8Hz
%   RctMax = 79;                % Maximum recruitment (in %MVC): 79%
%
%   % Start Simulation
%   [MP, SD, DD, NbofActiveMU, t, FR, DisPattern, sepMuEMG] = ...
%       SEMG_simulator_v4(LibraryName, Exitation, InSimPrd, fs, CV, varA, ...
%       DeltaE, MinFR, RctMax);
%   
%   % Plot
%   figure;
%   subplot(3,2,1); plot(t, Exitation); 
%       title('Exitation %MVC'); xlim([0 InSimPrd]);
%   subplot(3,2,3); plot(t, NbofActiveMU); 
%       title('Number of Active MU'); xlim([0 InSimPrd]);
%   subplot(3,2,5); bar(t, DisPattern(:,1)); 
%       title('Discharge MU #1'); xlim([0 InSimPrd]);
%   subplot(3,2,2); plot(t, MP(1,:)); 
%       title('sEMG sensor #1'); xlim([0 InSimPrd]);
%   subplot(3,2,4); plot(t, MP(2,:)); 
%       title('sEMG sensor #2'); xlim([0 InSimPrd]);
%   subplot(3,2,6); plot(t, sepMuEMG{1}(1,:)); 
%       title('separated MUAP #1,sensor #1'); xlim([0 InSimPrd]);
%
function [MP, SD, DD, NbofActiveMU, t, FR, DisPattern, sepMuEMG] = SEMG_simulator_v4_headless(LibraryName, Exitation, InSimPrd, fs, CV, varA, DeltaE, MinFR, RctMax)
    % Headless patch for PAPER_Multichannel_TVD.tex, Section sub:realistic-methods:
    %  (1) ProgressBar() below is a no-op (waitbar() below it crashes without
    %      a display).
    %  (2) The delay-injection loop originally read MU.MUAP{MU_Nb}(Channel,:)
    %      -- but MU was already `clear`ed earlier in this same function, so
    %      that line throws "MU undefined" whenever CV differs from the
    %      library's native 4 m/s (i.e. on every call this paper makes). Fixed
    %      to read the local MUAP{MU_Nb}(Channel,:) (the as-loaded, not-yet-
    %      delayed template) instead, matching the surrounding comment's
    %      stated intent of delaying the library's own per-channel MUAP.
    % Loading Library
    %	MU (MUAP database)
    %   |-> MU.PFR(k)        -> Peak Firing Rate of MUAP "k"
    %   |-> MU.RTE(k)        -> Exitation Threshold of MUAP "k"
    %   |-> MU.MUAP{k}(ch,t) -> MUAP Waveform in function of time
    %                            "t and channel "ch"
    MU = load(LibraryName);
    PeakFR = MU.PFR;
    RctThdE = MU.RTE;
    MUAP = MU.MUAP;
    Fs_sim = 1024;              % Sampling freq. (in Hz) of the simulator, used by the librairies.
    CV_sim = 4;                 % Original conduction velocity.
    clear MU;
    
    % Resampling factor of the MUAPs database
    Resamplingfactor = fs/(4*Fs_sim);

    % if Exitation is scalar, set a constant vector
    if length(Exitation)==1, 
        TimeLength = InSimPrd*fs;
        Exitation = Exitation*ones(1,TimeLength); 
    else
        TimeLength = length(Exitation);
    end

    % Maximal number of active MU over the whole simulation
    NbActivMU = size(find(max(Exitation)>=RctThdE),1);

	% Init de la progress bar
    ProgressBar(NbActivMU+TimeLength, 0, 'Firing Rate Computing', 'Simulation sEMG'); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. Compute Firing Rate for each active MU
        % Get matrix form
         E = Exitation(:)*ones(1,NbActivMU); 
         RctThdE = RctThdE(1:NbActivMU)*RctMax/max(RctThdE);
         RctThdE = (RctThdE(:)*ones(1,TimeLength))';
         maxPeakFR = max(PeakFR);
         PeakFR = PeakFR(1:NbActivMU);
         PeakFR = (PeakFR(:)*ones(1,TimeLength))';

        % Excitatory drive-firing rate relationship 
         ge = (maxPeakFR-MinFR)/(100-RctMax); 

        % Compute Firing Rate (FR) of each active MU at each time
         FR = (E>=RctThdE) .* (ge*(E-RctThdE) + MinFR);

        % Limit Firing Rate (FR) to PeakFR
         FR = FR - (FR>PeakFR).*(FR-PeakFR);
         clear E RTE;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2. Compute discharge Patern ( Dirac(t-k/FR-to) ) for each active MU
        T = zeros(1,NbActivMU);                     % Last discharge time for each MU
        NbofActiveMU = zeros(TimeLength,1);         % Number of active MU along t
        DisPattern = zeros(TimeLength,NbActivMU); 

        for it = 1 : TimeLength
            % Current simulation time (in s)
            t = (it-1)/fs*ones(1,NbActivMU);
            
            % Actualize Progress Bar
            if ProgressBar(NbActivMU+TimeLength, it, ['Firing Pattern t=' num2str(t(1)) 's'])
                disp('Stopped by user...');
                MP=[]; SD=[]; DD=[]; NbofActiveMU=[]; t=[]; FR=[]; DisPattern=[];
                ProgressBar;    % Delete ProgressBar
                return;
            end

            % Number of active MU at t
            NbActivMU_t = find(FR(it,:)==0,1)-1;
            if isempty(NbActivMU_t), NbActivMU_t=NbActivMU; end

            % If at least one active MU
            if NbActivMU_t
                Tdech = ones(1,NbActivMU_t)./FR(it,1:NbActivMU_t); % Discharge Period at t
                
                % Std-dev of jitter using percentage of Tdech (VarTo ~ 10 to 20%):
                %stddevTo = (VarTo/100)*Tdech = cv*Tdech;
                
                % Std-dev of jitter using Clamann's law : 
                stddevTo = 0.91 * Tdech.^2 + 0.004;
                
                To = stddevTo.*randn(1,NbActivMU_t); % Random variation of Tdech

                % Ones if MUAP fire for each active MU (Check duration between
                %   last disharge time T and now)
                ActiveFireVector = (t(1:NbActivMU_t) - T(1:NbActivMU_t)) >= (Tdech + To);

                % if a MUAP is fired, set actual time value to the vector T,
                %   with non-active MU (Modify only the values of "t" ??for which
                %   the bit mask "ResetVector" is set)
                ResetVector = ones(1,NbActivMU);
                ResetVector(1:NbActivMU_t) = ActiveFireVector;
                T = T - (T-t).*ResetVector;

                % and Set to 1 the corresponding MUAPs to the matrice
                %   DisPattern at the time t, for active MU only
                FireVector = zeros(1, NbActivMU);
                FireVector(1:NbActivMU_t) = ActiveFireVector;
                DisPattern(it, :) = FireVector;

                % Set the matrix NbofActiveMU with the number of active MU at
                %   this simulation time t
                NbofActiveMU(it) = NbActivMU_t;
            else
                % If no active MU set, for all MU, the current simulation time
                %   "t" to "T"
                T = t;
                NbofActiveMU(it) = 0;
            end
        end




    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3. Convolving MUAP waveform to the discharge Pattern
    NbOfChannels = size(MUAP{1},1);

    % PreAllocation of mono_sig, MP and if needed sepMuEMG
	if isequal(fs, 4*Fs_sim) % ie. if Resamplingfactor == 1
    	Smpl_MU_Len = length(MUAP{1}(1,:));
    elseif Resamplingfactor < 1
        Smpl_MU_Len = length(downsample(MUAP{1}(1,:), round(1/Resamplingfactor)));
    else
        Smpl_MU_Len = length(interp(MUAP{1}(1,:), round(Resamplingfactor)));
	end
    mono_sig = zeros(NbOfChannels, TimeLength+Smpl_MU_Len-1);
    MP = zeros(NbOfChannels, TimeLength);
    if nargout>=8, sepMuEMG = cell(NbActivMU,1); end
    
    % Main Loop
    for MU_Nb = 1 : NbActivMU % For each MU pointed by MU_Nb
        % Actualize Progress Bar
        if ProgressBar(NbActivMU+TimeLength, TimeLength+MU_Nb, ['MUAP ' num2str(MU_Nb) '/' num2str(NbActivMU) ' - Convolve and add'])
            disp('Stopped by user...');
            MP=[]; SD=[]; DD=[]; NbofActiveMU=[]; t=[]; FR=[]; DisPattern=[];
            ProgressBar;    % Delete ProgressBar
            return;
        end
        
        % Random Amplitude Modulation (stationnary)
        Mod = 1+(varA/100)*randn(TimeLength, 1);  
        
        % Delaying MUAPs with new Conduction velocity (in function paramters)
        if(~isequal(CV, CV_sim))  % if CV = original CV (4m/s), do not modify delays, else:
            % Delay factor relative to original CV (CV_sim = 4m/s)
            DelayFactor = -(4*Fs_sim*DeltaE/CV_sim) * (1-CV_sim/CV);

            % For each channels from 2 to 9, delay MUAP
            for Channel = 2 : NbOfChannels   
                % We create a delay by additing the difference between the asChanneled delay
                %	and the constant delay of 4 m.s-1 of the library of the simulator.
                %	DeltaE*Fe/(CV-4) is in sample.
                % -> Here the Fourier domain is used:
                MUAP{MU_Nb}(Channel,:) = Perfect_Delay(MUAP{MU_Nb}(Channel,:), (Channel-1)*DelayFactor);
            end
        end

        %*** Changes made by Fr�d�ric LECLERC, the 12th of July, 2007. ***%
        % For each channels, covolving Discharge train (dirac) with Loaded MUAP Waveform
        %   ->  mono_sig[Channel] = DischargeTrain * MUAPi[Channel]
        for Channel = 1 : NbOfChannels
            % Resampling MU by a factor of dependant of the sampling frequency
            % Fs.
            if isequal(fs, 4*Fs_sim) % ie. if Resamplingfactor == 1
                Smpl_MU = MUAP{MU_Nb}(Channel,:);
            elseif Resamplingfactor < 1
                Smpl_MU = downsample(MUAP{MU_Nb}(Channel,:), round(1/Resamplingfactor));
            else
                Smpl_MU = interp(MUAP{MU_Nb}(Channel,:), round(Resamplingfactor));
            end

            % Convolving discharge train with IAP template
            mono_sig(Channel,:) = conv(DisPattern(:,MU_Nb).*Mod, Smpl_MU);
        end

        % Summating new contribution of this active MUs
        MP = MP + mono_sig(:,1:TimeLength);
        
        % Separated sEMG
        if nargout>=8
            sepMuEMG{MU_Nb} = mono_sig(:,1:TimeLength);
        end
    end

    % Single-differential recording (SD)
    SD = diff(MP);

    % Double-differential recording (DD)                  
    DD = diff(SD);
    
    % Delete ProgressBar
    ProgressBar;    
    
    % Time Axis
    t = 0:1/fs:(TimeLength-1)/fs;
    
end




% Progress Bar
function cancel = ProgressBar(K, k, Message, title)
    % Headless no-op: the original body called waitbar()/getappdata(), which
    % crash with no display. Disabled entirely per PAPER_Multichannel_TVD.tex,
    % Section sub:realistic-methods ("its GUI progress-bar calls, which crash
    % without a display, were disabled").
    cancel = false;
end