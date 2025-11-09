% EEG Signal Classification – Extended Version
% Author: Jiawen Liu
% Date: November.9th 2025

clear; clc; close all;

% Parameters
fs = 250;                % Sampling frequency (Hz)
t = 0:1/fs:2;            % 2-second signal
nSamples = 210;          % Total samples (70 per class)
freqBands = [10, 20, 40];% alpha, beta, gamma frequencies

% Generate synthetic EEG-like signals 
simulateEEG = @(freq) sin(2*pi*freq*t) + 0.5*randn(size(t));

signals = zeros(nSamples, length(t));
labels = zeros(nSamples,1);

for i = 1:nSamples
    class = ceil(i/70);
    freq = freqBands(class);
    signals(i,:) = simulateEEG(freq);
    labels(i) = class;
end

% Bandpass filtering (8–45 Hz)
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder', 4, ...
    'HalfPowerFrequency1', 8, ...
    'HalfPowerFrequency2', 45, ...
    'SampleRate', fs);

fSignals = zeros(size(signals));
for i = 1:nSamples
    fSignals(i,:) = filtfilt(bpFilt, signals(i,:));
end

% Feature extraction
% Features: mean, variance, total power, dominant freq, power ratio (alpha/beta range)
features = zeros(nSamples,5);
freqRange = 0:fs/length(t):fs/2;

for i = 1:nSamples
    sig = fSignals(i,:);
    features(i,1) = mean(sig);
    features(i,2) = var(sig);
    [pxx,f] = pwelch(sig,[],[],[],fs);
    features(i,3) = bandpower(pxx,f,[8 45],'psd');
    
    % Dominant frequency
    [~,idx] = max(pxx);
    features(i,4) = f(idx);
    
    % Power ratio (8–15 Hz / 16–30 Hz)
    alphaPower = bandpower(pxx,f,[8 15],'psd');
    betaPower  = bandpower(pxx,f,[16 30],'psd');
    features(i,5) = alphaPower / (betaPower + 1e-6);
end

% Train/test split
cv = cvpartition(labels,'HoldOut',0.2);
Xtrain = features(training(cv),:);
Ytrain = labels(training(cv));
Xtest  = features(test(cv),:);
Ytest  = labels(test(cv));

% Model 1 – SVM classification
SVMModel = fitcecoc(Xtrain,Ytrain);  % multi-class SVM
Ypred_svm = predict(SVMModel,Xtest);
acc_svm = mean(Ypred_svm == Ytest);
fprintf('SVM Classification Accuracy: %.2f%%\n', acc_svm*100);

% Model 2 – MLP Neural Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
net.trainParam.showWindow = false; % hide GUI
net = train(net, Xtrain', full(ind2vec(Ytrain')));
Ypred_mlp = vec2ind(net(Xtest'));
acc_mlp = mean(Ypred_mlp' == Ytest);
fprintf('MLP Classification Accuracy: %.2f%%\n', acc_mlp*100);

%Confusion matrix
figure;
confusionchart(Ytest, Ypred_mlp);
title('Confusion Matrix (MLP Classifier)');

% 3D Feature Visualization
figure;
scatter3(features(:,2), features(:,3), features(:,4), 40, labels, 'filled');
xlabel('Variance'); ylabel('Power (8–45Hz)'); zlabel('Dominant Freq (Hz)');
title('3D Feature Distribution of Simulated EEG Signals');
grid on;

% Plot example signals and PSD
figure;
plot(t, signals(1,:), 'b'); hold on;
plot(t, signals(71,:), 'r');
plot(t, signals(141,:), 'g');
title('Simulated EEG Signals (α, β, γ)');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Alpha (10Hz)','Beta (20Hz)','Gamma (40Hz)');

figure;
[pxx,f] = pwelch(signals(1,:),[],[],[],fs);
plot(f,10*log10(pxx));
title('Power Spectral Density Example');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');

