%% Main program for the simulation of the effect of external noise in NObSP
%% using SVM for dynamic systems.

% This program evaluate the impact of external noise using the values for N
% and m that produced the lowest RMSE, the robustness of the proposed 
% algorithm was evaluated by changing the signal-to-noise ratio (SNR) of 
% the output signal. We perform simulations for values of SNR ranging from 
% 0.8 dB up to 18 dB.
% The impact of changing the amplitude of noise in the output was computed 
% for the fit of the model, as well as for the estimated projections. The
% impact was also computed for the estimation of the output obtained using
% the alpha coefficients computed in (13).

%% Clearing enviroment and preparing data

clear
close all
clc
P=[500:100:1900 2000:500:10000]; % Vector of number of samples N
M=[5:5:20 30:10:80 100:20:200]; % Vector of model order m
Sn = [0.01 0.10 0.25 0.35 0.5]; % External noise
Forend=15; % Number of iterations for train

for Sim=1:2 % The first simulation evaluate the performance of the Wiener 
    % structure and the second for the Hammerstein

    % Define the value of N and m
    if Sim==1
        N_Tit = 'Wiener Structure';
        M_n = 4;
        n = 26;
    else
        N_Tit = 'Hammerstein Structure';
        M_n = 4;
        n = 5;
    end

    % Variable space
    Model_Tr = cell(length(Sn),Forend,4);
    M_Err_Tr_Fit = zeros(length(Sn),Forend);
    M_Err_Tr_Proy = zeros(length(Sn),Forend,2);
    M_Err_Te_Fit = zeros(length(Sn),Forend,2);
    M_Err_Te_Proy = zeros(length(Sn),Forend,2);
    M_Err_Te_Alp = zeros(length(Sn),Forend,2);
    M_Ti_Te = zeros(length(Sn),Forend,2);
    
    for Cont=1:length(Sn) % External noise
        N=P(n);
        m = M(M_n);
        %% TRAIN STAGE
        for Conti=1:Forend % Number of iterations
            noise = wgn(N,1,Sn(Cont),'linear'); % Noise signal

            % PRBS signal
            base=prbs(99,100,double(rand(1,99)>.5));
            base_In=ones(1,N);
            for i=1:100
            base_In(((i-1)*(N/100))+1:((i)*(N/100)))=base(i);
            end

            X=[(30*base_In')-15 15*sin(21*pi*linspace(0,1,N)')]; % Input signal
    
            if Sim==1
                % Parameters of Wiener structure
                C1=(1.5e-4);
                C2=(.5);

                b1 = [0.0089 -0.0045 -0.0045 0.0089];
                a1 = [1 -2.5641 2.2185 -0.6456];
                P1 = filter(b1,a1,X(:,1));
                Y1 = C1*(P1.^3);
               
                b2 = [0.0047 -0.0142 -0.0142 0.0047];
                a2 = [1 -2.458 2.262 -0.7654];
                P2 = filter(b2,a2,X(:,2));
                Y2 = C2*(sinc(P2).*P2.^2);
            
            else
                % Parameters of Hammerstein structure
                C1=(3e-4);
                C2=(.75);

                P1 = X(:,1).^3;
                b1 = [0.0089 -0.0045 -0.0045 0.0089];
                a1 = [1 -2.5641 2.2185 -0.6456];
                Y1 = C1*filter(b1,a1,P1);
            
                P2 = sinc(X(:,2)).*X(:,2).^2;
                b2 = [0.0047 -0.0142 -0.0142 0.0047];
                a2 = [1 -2.458 2.262 -0.7654];
                Y2 = C2*filter(b2,a2,P2);
            end

            Sys = Y1 + Y2 + noise; % Compound System
            [input,output] = prepareData(X,Sys,m); % Preparing the data for the regression model.
        
            %% Fitting the model
            MdlGau = fitrsvm(input,output,'Standardize',true,'KernelFunction','gaussian'); % Fitting a SVM with RBF kernel
            outfit = predict(MdlGau,input); % Finding the predicted output of the model
            error = Sys(m:end)-outfit; % Fitting
            error_MAE = sum(abs(error))/(N*abs(max(Sys(m:end))-min(Sys(m:end)))); % Fitting error
            M_Err_Tr_Fit(Cont,Conti,1) = error_MAE;
            %% Preparing the input data for the decompòsition using NObSP
            
            inputPrepro = (input-MdlGau.Mu)./MdlGau.Sigma;
            inputSV = inputPrepro(MdlGau.IsSupportVector,:); % Extracting the support vectors
            [Y,Model_Tr{Cont,Conti,4}] = NObSP_Tot(input,inputSV,m,MdlGau);
            error = (Y1(m:end)-mean(Y1(m:end)))-(Y(:,1)-mean(Y(:,1)));
            M_Err_Tr_Proy(Cont,Conti,1,1)=sum(abs(error))/(N*abs(max(Y1(m:end))-min(Y1(m:end))));
            error = (Y2(m:end)-mean(Y2(m:end)))-(Y(:,2)-mean(Y(:,2)));
            M_Err_Tr_Proy(Cont,Conti,2,1)=sum(abs(error))/(N*abs(max(Y2(m:end))-min(Y2(m:end))));
            Model_Tr{Cont,Conti,3} = Y;
            Model_Tr{Cont,Conti,1} = MdlGau;
            Model_Tr{Cont,Conti,2} = inputSV;
        end
        
        %% TEST STAGE
        N2=N; % Select the number of iterations
        SM = randi([1,Forend]); % Define the model in train stage

        % Variable space
        MdlGau_T = Model_Tr{Cont,SM,1};
        XSV = Model_Tr{Cont,SM,2};
        Alpha = Model_Tr{Cont,SM,4};
        ForendTe=Forend;
            
        for Conti_te=1:ForendTe % Number of iterations

            noise = wgn(N,1,Sn(Cont),'linear'); % Noise signal

            % PRBS signal
            baset=prbs(99,100,double(rand(1,99)>.5));
            baset_In=ones(1,N2);
            for i=1:100
                baset_In(((i-1)*(N2/100))+1:((i)*(N2/100)))=baset(i);
            end

            % Input signal
            In=rand();
            Xn=[((30*baset_In)-15)' 15*sin(21*pi*linspace(In,1+In,N2)')];
            
            if Sim==1
                % Parameters of Wiener Structure
                C1=(2e-4);
                C2=(.50);

                b1 = [0.0089 -0.0045 -0.0045 0.0089];
                a1 = [1 -2.5641 2.2185 -0.6456];
                P1t = filter(b1,a1,Xn(:,1));
                Y1t = C1*(P1t.^3);
               
                b2 = [0.0047 -0.0142 -0.0142 0.0047];
                a2 = [1 -2.458 2.262 -0.7654];
                P2t = filter(b2,a2,Xn(:,2));
                Y2t = C2*(sinc(P2t).*P2t.^2);
            else
                % Parameters of Hammerstein Structure
                C1=(2e-4);
                C2=(.75);

                P1t = Xn(:,1).^3;
                b1 = [0.0089 -0.0045 -0.0045 0.0089];
                a1 = [1 -2.5641 2.2185 -0.6456];
                Y1t = C1*filter(b1,a1,P1t);
            
                P2t = sinc(Xn(:,2)).*Xn(:,2).^2;
                b2 = [0.0047 -0.0142 -0.0142 0.0047];
                a2 = [1 -2.458 2.262 -0.7654];
                Y2t = C2*filter(b2,a2,P2t);
            end

            y_nx = [Y1t Y2t]; % Input signal (error calculation)
            y_n = Y1t+Y2t+wgn(N,1,Sn(Cont),'linear'); % Compound signal with noise
            [input_t,output_t] = prepareData(Xn,y_n,m);
            Xn_prepro = (input_t-MdlGau_T.Mu)./MdlGau_T.Sigma; % Preprocessing the data
            tic
            Ypn_proy = NObSP(input_t,XSV,m,MdlGau_T); % Calculate the retreive contributions
            M_Ti_Te(Cont,Conti_te,1) = toc; % Time of NObSP by projections
            fprintf('en -> %d\n',M_Ti_Te(Cont,Conti_te,1))
            fprintf('Test (Alpha) - ')
            tic
            dis_X = diag(Xn_prepro*Xn_prepro')-2*Xn_prepro*XSV'+ones(N-m+1,1)*...
                (diag(XSV*XSV'))'; % Computing the euclidean distance for the construction of the kernel
            K = exp(-1*dis_X./MdlGau_T.KernelParameters.Scale); % Calculating the kernel matrix
            Ypn = K*Alpha; % Estimating the output for the nonlinear contributions of each input regressor
            M_Ti_Te(Cont,Conti_te,2) = toc; % Time of NObSP by coefficients
            fprintf('en -> %d\n',M_Ti_Te(Cont,Conti_te,2))
            outfit_n = predict(MdlGau_T,input_t); % Finding the predicted output of the model
            % Calculate the fit error
            error = y_n(m:end)-outfit_n;
            M_Err_Te_Fit(Cont,Conti_te,1)=sum(abs(error))/(N*abs(max(y_n(m:end))-min(y_n(m:end))));
            % Calculate the error of each contribution by projections
            error = (y_nx(m:end,1)-mean(y_nx(m:end,1)))-(Ypn_proy(:,1)-mean(Ypn_proy(:,1)));
            M_Err_Te_Proy(Cont,Conti_te,1)=sum(abs(error))/(N*abs(max(y_nx(m:end,1))-min(y_nx(m:end,1))));
            error = (y_nx(m:end,2)-mean(y_nx(m:end,2)))-(Ypn_proy(:,2)-mean(Ypn_proy(:,1)));
            M_Err_Te_Proy(Cont,Conti_te,2)=sum(abs(error))/(N*abs(max(y_nx(m:end,2))-min(y_nx(m:end,2))));
            % Calculate the error of each contribution by coefficients
            error = (y_nx(m:end,1)-mean(y_nx(m:end,1)))-(Ypn(:,1)-mean(Ypn(:,1)));
            M_Err_Te_Alp(Cont,Conti_te,1)=sum(abs(error))/(N*abs(max(y_nx(m:end,1))-min(y_nx(m:end,1))));
            error = (y_nx(m:end,2)-mean(y_nx(m:end,2)))-(Ypn(:,2)-mean(Ypn(:,1)));
            M_Err_Te_Alp(Cont,Conti_te,2)=sum(abs(error))/(N*abs(max(y_nx(m:end,2))-min(y_nx(m:end,2))));
        end

        disp('ERROR IN TEST:')
        disp('Outfit -> ')
        fprintf('MAE -> %d\n',mean(M_Err_Te_Fit(Cont,:)))
        disp('PROJECTIONS')
        disp('Cuadratic-> ')
        fprintf('MAE -> %d\n',mean(M_Err_Te_Proy(Cont,:,1)))
        disp('Sinc with Cubic-> ')
        fprintf('MAE -> %d\n',mean(M_Err_Te_Proy(Cont,:,2)))
        fprintf('In %d s\n\n',mean(M_Ti_Te(Cont,:,1)))
        disp('ALPHA COEF')
        disp('Cuadratic-> ')
        fprintf('MAE -> %d\n',mean(M_Err_Te_Alp(Cont,:,1)))
        disp('Sinc with Cubic-> ')
        fprintf('MAE -> %d\n',mean(M_Err_Te_Alp(Cont,:,2)))
        fprintf('In %d s\n\n',mean(M_Ti_Te(Cont,:,2)))
        
    end
end
