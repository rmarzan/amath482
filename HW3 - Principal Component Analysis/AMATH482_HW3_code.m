clear all; close all; clc;

%%% Name: ROSEMICHELLE MARZAN
%%% Course: AMATH 482
%%% Homework 3, Due 2/21/2019

%%  Data Acquisition

% import all recordings
load('cam1_1.mat'); load('cam2_1.mat'); load('cam3_1.mat')
load('cam1_2.mat'); load('cam2_2.mat'); load('cam3_2.mat')
load('cam1_3.mat'); load('cam2_3.mat'); load('cam3_3.mat')
load('cam1_4.mat'); load('cam2_4.mat'); load('cam3_4.mat')

% % inspect one video at a time
% implay(vidFrames1_4)
% % inspect one frame to get limits for cropping area
% BW = rgb2gray(vidFrames1_4(:,:,:,392));
% imshow(BW)

cam1_1 = trackCan(vidFrames1_1, 250, 300, 400, 0, 480);
cam2_1 = trackCan(vidFrames2_1, 250, 240, 340, 0, 480);
cam3_1 = trackCan(vidFrames3_1, 250, 0, 640, 290, 460);
cam1_2 = trackCan(vidFrames1_2, 250, 290, 640, 0, 480);
cam2_2 = trackCan(vidFrames2_2, 250, 0, 640, 0, 480);
cam3_2 = trackCan(vidFrames3_2, 245, 0, 640, 200, 340);
cam1_3 = trackCan(vidFrames1_3, 250, 300, 400, 0, 480);
cam2_3 = trackCan(vidFrames2_3, 250, 220, 430, 0, 480);
cam3_3 = trackCan(vidFrames3_3, 250, 0, 640, 140, 350);
cam1_4 = trackCan(vidFrames1_4, 245, 330, 430, 215, 480);
cam2_4 = trackCan(vidFrames2_4, 250, 200, 430, 0, 480);
cam3_4 = trackCan(vidFrames3_4, 235, 250, 640, 145, 300);


%% Principal Component Analysis

case1 = readmatrix('case1_A.csv');
[u1,s1,v1] = svd(case1); 
case2 = readmatrix('case2_A.csv');
[u2,s2,v2] = svd(case2);
case3 = readmatrix('case3_A.csv');
[u3,s3,v3] = svd(case3);
case4 = readmatrix('case4_A.csv');
[u4,s4,v4] = svd(case4);
sig1 = diag(s1); sig2 = diag(s2); sig3 = diag(s3); sig4 = diag(s4);

figure(1) 
% plot the principal components
subplot(4,2,1)
svd_proj_1 = u1'*case1;
plot(1:length(case1(1,:)),svd_proj_1(1:3,:))
legend('PC1', 'PC2', 'PC3')
xlabel('Time')
ylabel('Relative Displacement')
title('Case 1')

subplot(4,2,3)
svd_proj_2 = u2'*case2;
plot(1:length(case2(1,:)),svd_proj_2(1:3,:))
legend('PC1', 'PC2', 'PC3')
xlabel('Time')
ylabel('Relative Displacement')
title('Case 2')

subplot(4,2,5)
svd_proj_3 = u3'*case3;
plot(1:length(case3(1,:)),svd_proj_3(1:3,:))
legend('PC1', 'PC2', 'PC3')
xlabel('Time')
ylabel('Relative Displacement')
title('Case 3')

subplot(4,2,7)
svd_proj_4 = u4'*case4;
plot(1:length(case4(1,:)),svd_proj_4(1:3,:))
legend('PC1', 'PC2', 'PC3')
xlabel('Time')
ylabel('Relative Displacement')
title('Case 4')

% plot the energies of each mode for each case
cases = [sig1, sig2, sig3, sig4];
for i = 1:4
    subplot(4,2,2*i)
    plot(cases(:,i).^2/sum(cases(:,i).^2),'k*','Linewidth',2)
    ylabel('Energy')
    set(gca,'Xtick', [1:6])
    xlabel('Principal Component')
    title(['Case ', int2str(i)])
    for j = 1:6
        val = cases(j,i)^2/sum(cases(:,i).^2);
        text(j,val,num2str(val));
    end
end


%% FUNCTIONS

function points = trackCan(video, threshold, xmin, xmax, ymin, ymax)
    nFrames = size(video,4);     % total # of frames
    points = zeros(2,nFrames);   % paint can coordinates for all frames
    for i = 1:nFrames
        keep = [];               % collects white areas in frame
        % convert the frame to grayscale
        BW = rgb2gray(video(:,:,:,i));
        for j = 1:length(BW(:))  % inspect each pixel
            % convert pixel # to (X,Y) coordinate
            [Y,X] = ind2sub([480,640],j); 
            % is pixel in ROI?
            if X > xmin && X < xmax && Y > ymin && Y < ymax
                % is pixel at or above white-ness threshold?
                if BW(Y,X) >= threshold                             
                    keep = [keep; X, Y];
                end
            end
        end
        B = rmoutliers(keep);     % remove white pixels not part of paint can
        % get coordinate of approximate center-of-mass
        x = mean(B(:,1));               
        y = mean(B(:,2));
        points(:,i) = [x;y];      % save coordinate
    end
end