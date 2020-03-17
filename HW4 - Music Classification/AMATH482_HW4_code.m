clear all; close all; clc;

%%% Name: ROSEMICHELLE MARZAN
%%% Course: AMATH 482
%%% Homework 4, Due 3/6/2019

%% Training

files = dir('*.mp3');
numSongs = 90;
% train rows: arbitrary #; all clips have slightly varying # of data points
% depending on recording duration
train = zeros(230000,numSongs);     
for k = 1:numSongs
    A = audioread(files(k).name);
    train(1:length(A),k) = A;
end

Fs = 44100;     % sampling rate
L = 5;          % want 5 s songs
train = train(1:(Fs*L),:);    % truncate training data
train = train(1:4:end,:); % take every fourth point

% Create spectrograms
n = length(train3(:,1));           % # of data points/clip
t = linspace(0,L,n);               % vector of time points
a = 1000; dtau = 0.05;             % sliding filter parameters
tslide = 0:dtau:L;
spec = zeros(length(tslide),n,numSongs);    % matrix of spectrograms
for i = 1:numSongs
    for j = 1:length(tslide)
        g = exp(-a*(t - tslide(j)).^2);
        Sg = g.*(train3(:,i).');
        Sgt = fft(Sg); 
        spec(j,:,i) = fftshift(abs(Sgt));
    end
end

% separate matrix by classes
% 5567625 = # of data points * tslide length
class1spec = reshape(spec(:,:,1:30), [5567625,30]);
class2spec = reshape(spec(:,:,31:60), [5567625,30]);
class3spec = reshape(spec(:,:,61:90), [5567625,30]);

feature = 5;
[U,S,V,w,v1,v2,v3] = trainer(class1spec,class2spec,class3spec,feature)

%% Classification

clc;
% load test songs (.mat file)
load exp1testsongs.mat
numTestSongs = 30;

% get spectrogram
a = 1000; dtau = 0.05;
tslide = 0:dtau:L;
test_spec = zeros(length(tslide),n,numTestSongs);
for i = 1:numTestSongs
    for j = 1:length(tslide)
        g = exp(-a*(t - tslide(j)).^2);
        Sg = g.*(test1(:,i).');
        Sgt = fft(Sg); 
        test_spec(j,:,i) = fftshift(abs(Sgt));
    end
end
test_spec = reshape(test_spec(:,:,:), [5567625,numTestSongs]);

% PCA projection
test_pca = zeros(90,30);
for i = 1:30
    test_pca(:,i) = U'*test_spec(:,i);
end

% LDA projection
test_lda = zeros(1,30);
for i = 1:30
    test_lda(1,i) = w'*test_pca(1:feature,i);
end

% mean of projected training points
m1_proj = mean(v1);
m2_proj = mean(v2);
m3_proj = mean(v3);

% Mean-based thresholding/classification
for i = 1:30
    distances = [abs(m1_proj - test_lda(i));
                abs(m2_proj - test_lda(i));
                abs(m3_proj - test_lda(i))];
    [~,index] = min(distances);
    
    % CHANGE LABELS
    if index == 1
        disp('Class 1')
    elseif index == 2
        disp('Class 2')
    else
        disp('Class 3')
    end
end


function [U,S,V,w,v1,v2,v3] = trainer(class1spec,class2spec,class3spec,feature)
    n1 = size(class1spec,2); 
    n2 = size(class2spec,2);
    n3 = size(class3spec,2);
    
    [U,S,V] = svd([class1spec, class2spec, class3spec],'econ');
    
    allsongs = S*V'; % projection onto principal components
    class1 = allsongs(1:feature,1:n1);
    class2 = allsongs(1:feature,n1+1:n1+n2);
    class3 = allsongs(1:feature,n1+n2+1:n1+n2+n3);
    
    m1 = mean(class1,2);
    m2 = mean(class2,2);
    m3 = mean(class3,2);
    mu = mean([class1,class2,class3],2);
    
    % within-class scatter matrix
    Sw = 0; 
    for k=1:30
        Sw = Sw + (class1(:,k)-m1)*(class1(:,k)-m1)';
    end
    for k=1:30
        Sw = Sw + (class2(:,k)-m2)*(class2(:,k)-m2)';
    end
    for k=1:30
        Sw = Sw + (class3(:,k)-m3)*(class3(:,k)-m3)';
    end
    
    % between-class scatter matrix
    Sb = ((m1-mu)*(m1-mu)' + (m2-mu)*(m2-mu)' + (m3-mu)*(m3-mu)');

    % linear discriminant analysis: maximize Sb, minimize Sw
    [V1,D1] = eig(Sb,Sw);
    [~,ind] = max(abs(diag(D1))); % w = eigenvec of largest eigenval
    w = V1(:,ind); w = w/norm(w,2);

    % LDA-projected class matrices
    v1 = w'*class1;
    v2 = w'*class2;
    v3 = w'*class3; 
end