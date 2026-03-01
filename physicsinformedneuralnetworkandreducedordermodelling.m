%% PINN-ROM (10000 EPOCHS)
% Physics-Informed Neural Network & Reduced Order Model

%% Clearing console, ensuring no leftover variables and figures
clear; clc; close all;

%% 1. SYNTHETIC PERIODIC DATASET GENERATION WITH ADDITIVE NOISE
N = 20000;
t = linspace(0,50,N)';           % time vector
dt = t(2)-t(1);                  % time step

TEMP_clean = -10.23 + 4.67*sin(2*pi*t/24) + 2.5*sin(2*pi*t/12);
WS_clean   = 9.84 + 3.12*cos(2*pi*t/24) + 1.8*sin(2*pi*t/6);

TEMP = TEMP_clean + 0.001*randn(N,1);
WS   = WS_clean   + 0.001*randn(N,1);

X = [TEMP WS];                   % combine as features

%% 2. COLUMN-WISE Z-SCORE NORMALIZATION
X_mean = mean(X,1);
X_std  = std(X,0,1);
X_norm = (X - X_mean) ./ X_std;

%% 3. REDUCED ORDER MODELLING SVD
[U,S,~] = svd(X_norm,'econ');
sing_vals = diag(S);
energy = cumsum(sing_vals.^2)/sum(sing_vals.^2)*100;
K = find(energy>=99.99,1);
U = U(:,1:K);                    % retain dominant ROM modes
fprintf("K = %d (%.4f%% energy)\n",K,energy(K));

%% 4. 80:20 SPLIT AND PREPARE INPUTS (INCLUDE TIME AS FEATURE)
train_n = round(0.8*N);
X_train = X_norm(1:train_n,:);
U_train = U(1:train_n,:);        % ROM modes as input
X_test  = X_norm(train_n+1:end,:);
U_test  = U(train_n+1:end,:);
t_train = t(1:train_n);          % time for train
t_test  = t(train_n+1:end);      % time for test

% Concatenate time as an extra input feature so the network output depends on t
U_train_in = [U_train, t_train];   % [train_n x (K+1)]
U_test_in  = [U_test,  t_test];    % [N-train_n x (K+1)]

% Convert to dlarray WITHOUT labels for standard matrix multiply
U_train_dl = dlarray(U_train_in);   % [batch x features]
X_train_dl = dlarray(X_train);      % [batch x outputs]
U_test_dl  = dlarray(U_test_in);    % for testing later (dlarray unformatted)

%% 5. NETWORK SIZE AND XAVIER-LIKE INITIALIZATION
input_size = K+1;                % +1 for the time feature
hidden1 = 128;
hidden2 = 64;
output_size = 2;

W1 = dlarray(randn(input_size,hidden1)*sqrt(2/(input_size+hidden1)));
b1 = dlarray(zeros(1,hidden1));
W2 = dlarray(randn(hidden1,hidden2)*sqrt(2/(hidden1+hidden2)));
b2 = dlarray(zeros(1,hidden2));
W3 = dlarray(randn(hidden2,output_size)*sqrt(2/(hidden2+output_size)));
b3 = dlarray(zeros(1,output_size));

%% 6. TRAINING CONFIGURATION AND ADAM INITIALIZATION
lr = 0.001;
epochs = 15000;
lambda = 1000;
omega = 2*pi/24;

trailingAvg_W1 = []; trailingAvgSq_W1 = [];
trailingAvg_b1 = []; trailingAvgSq_b1 = [];
trailingAvg_W2 = []; trailingAvgSq_W2 = [];
trailingAvg_b2 = []; trailingAvgSq_b2 = [];
trailingAvg_W3 = []; trailingAvgSq_W3 = [];
trailingAvg_b3 = []; trailingAvgSq_b3 = [];

loss_history = zeros(epochs,1);

%% 7. TRAINING LOOP
for epoch = 1:epochs

    [loss,gradients] = dlfeval(@modelLoss, ...
        W1,b1,W2,b2,W3,b3, ...
        U_train_dl,X_train_dl,omega,lambda);

    % Update parameters using adamupdate (separate states)
    [W1,trailingAvg_W1,trailingAvgSq_W1] = adamupdate(W1,gradients.W1,trailingAvg_W1,trailingAvgSq_W1,epoch,lr);
    [b1,trailingAvg_b1,trailingAvgSq_b1] = adamupdate(b1,gradients.b1,trailingAvg_b1,trailingAvgSq_b1,epoch,lr);
    [W2,trailingAvg_W2,trailingAvgSq_W2] = adamupdate(W2,gradients.W2,trailingAvg_W2,trailingAvgSq_W2,epoch,lr);
    [b2,trailingAvg_b2,trailingAvgSq_b2] = adamupdate(b2,gradients.b2,trailingAvg_b2,trailingAvgSq_b2,epoch,lr);
    [W3,trailingAvg_W3,trailingAvgSq_W3] = adamupdate(W3,gradients.W3,trailingAvg_W3,trailingAvgSq_W3,epoch,lr);
    [b3,trailingAvg_b3,trailingAvgSq_b3] = adamupdate(b3,gradients.b3,trailingAvg_b3,trailingAvgSq_b3,epoch,lr);

    loss_history(epoch) = extractdata(loss);  % store numeric loss

    if mod(epoch,1000)==0
        fprintf("Epoch %d | Loss: %.6e\n",epoch,loss_history(epoch))
    end
end

%% 8. TESTING / PREDICTION
% Forward pass on test inputs (U_test_dl prepared earlier)
X_pred_test = tanh(U_test_dl * W1 + b1);
X_pred_test = tanh(X_pred_test * W2 + b2);
X_pred_test = X_pred_test * W3 + b3;

% Extract numeric data for denormalization and metrics
X_pred_test = extractdata(X_pred_test);   % convert dlarray -> numeric
X_pred_test_denorm = X_pred_test .* X_std + X_mean;
X_test_denorm = X_test .* X_std + X_mean;

% Compute RMSE
rmse = sqrt(mean((X_test_denorm - X_pred_test_denorm).^2,'all'));
fprintf('PINN-ROM Test RMSE: %.8f\n',rmse);
target = 0.000478;

%% 9. PLOTS
figure('Position',[50 50 1400 600]);
% 9a. Loss convergence
subplot(1,2,1);
semilogy(1:epochs,loss_history,'LineWidth',2);
xlabel('Epoch'); ylabel('Loss'); title('Loss Convergence');
grid on;

% 9b. RMSE comparison
subplot(1,2,2);
bar([target, rmse],'FaceColor','flat');
set(gca,'XTickLabel',{'Target','PINN-ROM'});
ylabel('RMSE'); title('RMSE Comparison'); grid on;

%% 10. TEMP & WIND PREDICTION PLOT
figure('Position',[50 50 1400 500]);
subplot(2,1,1);
plot(t_test,X_test_denorm(:,1),'b','LineWidth',1); hold on;
plot(t_test,X_pred_test_denorm(:,1),'r--','LineWidth',1);
title('Temperature Prediction'); legend('True','PINN-ROM'); grid on;
xlabel('Time');
ylabel('Temperature');

subplot(2,1,2);
plot(t_test,X_test_denorm(:,2),'b','LineWidth',1); hold on;
plot(t_test,X_pred_test_denorm(:,2),'r--','LineWidth',1);
title('Wind Speed Prediction'); legend('True','PINN-ROM'); grid on;
xlabel('Time');
ylabel('Wind Speed');

%% ============================================
%% MODEL LOSS FUNCTION (LOCAL FUNCTION AT END)
%% ============================================
function [loss,gradients] = modelLoss(W1,b1,W2,b2,W3,b3,U_in,X_true,omega,lambda)
    % U_in : dlarray unformatted [batch x (K+1)], last column = time
    % Forward pass
    Z1 = tanh(U_in * W1 + b1);      % [batch x hidden1]
    Z2 = tanh(Z1 * W2 + b2);        % [batch x hidden2]
    X_pred = Z2 * W3 + b3;          % [batch x output_size]

    % Data loss (MSE)
    data_loss = mean((X_pred - X_true).^2,'all');

    % Time column used for AD
    t_col = U_in(:, end);           % [batch x 1] dlarray

    % For gradient computation, we need a scalar output
    % Sum all predictions to get a scalar
    X_pred_sum = sum(X_pred, 'all');
    
    % First derivative of scalar output w.r.t. time
    dXdt = dlgradient(X_pred_sum, t_col, 'EnableHigherDerivatives', true); % [batch x 1]
    
    % Sum of first derivatives (to get scalar for second derivative)
    dXdt_sum = sum(dXdt, 'all');
    
    % Second derivative w.r.t. time
    d2Xdt2 = dlgradient(dXdt_sum, t_col, 'EnableHigherDerivatives', true); % [batch x 1]

    % For physics residual, we need per-sample values
    % Compute per-sample sum of outputs
    s_per_sample = sum(X_pred, 2);  % [batch x 1]
    
    % Scale the second derivative to match per-sample scale
    % This is an approximation - we're distributing the total second derivative
    batch_size = size(U_in, 1);
    d2Xdt2_per_sample = d2Xdt2 / batch_size;  % Distribute evenly
    
    % Physics residual using per-sample values
    physics_res = d2Xdt2_per_sample + (omega^2) .* s_per_sample;   % [batch x 1]
    physics_loss = mean(physics_res.^2, 'all');

    % Total loss
    loss = data_loss + lambda * physics_loss;

    % Gradients w.r.t network parameters
    gradients = dlgradient(loss, struct('W1',W1,'b1',b1,'W2',W2,'b2',b2,'W3',W3,'b3',b3));
end