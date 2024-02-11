function SymDemo

% read image

[FileName,PathName] = uigetfile('*.bmp');
I = double(imread([PathName '\' FileName]));
% normalize between 0 and 255
I(:) = (I - min(I(:)))*255/(max(I(:)) - min(I(:)));


% detect skull
[M,I]=skull_detect(I);
figure, imagesc(M), colormap(gray), axis image; drawnow;
% return
h = size(I,1);
STATS = regionprops(M,'all');
midx = round(STATS.Centroid(1));
M = logical(M);

% display
subplot(2,2,1); imagesc(I),colormap(gray),axis image,
title('MR Image'); drawnow;
figure(1), subplot(2,2,2); imagesc(I), colormap(gray), axis image; drawnow;
hold on, plot([midx midx],[1 h], 'linewidth', 2); drawnow;
[b_x,b_y] = find(bwperim(M)== 1);
hold on, plot(b_y,b_x, '.c'); drawnow;

% creating images and masks
Im = I(:,midx:-1:1); % left image, we call this original/test image
ImMask  = M(:,midx:-1:1);% mask for original image
RefI = I(:,midx:end);% reference image, here it is the right side
RefIMask = M(:,midx:end);% mask for reference image

% start of the vertical scan and end of the vertical scan
starti=round(STATS.BoundingBox(2));
endi=round(STATS.BoundingBox(2) + STATS.BoundingBox(4));

% Top-down search: Computing the Bhattacharya coefficient-based score function
fact = 16; % histogram binsize, an important parameter
BC_diff_TD = score(Im,RefI,ImMask,RefIMask,starti,endi,fact);

figure(1), subplot(2,2,3),plot(starti:endi,BC_diff_TD);
title('Score plot for vertical direction');
set(gcf, 'color', [1 1 1]);

vert_scale = 30; % scale for finding maxima and minima of the vertical score function
[topy1, downy1]= find_largest_decreasing_segment(BC_diff_TD,vert_scale);

topy  = topy1(1);
downy = downy1(1);
% plot
subplot(2,2,3), hold on; plot(topy+ starti-1,BC_diff_TD(topy), 'r.',downy+ starti-1,BC_diff_TD(downy),'m.','MarkerSize',10);
%     text(round((topy+ starti-1+downy+ starti-1)/2),(BC_diff_TD(topy) + BC_diff_TD(downy))/2, num2str(0),'FontSize',10,'color','r');
topy = topy + starti-1;
downy = downy + starti-1;
figure(1), hold on,

% left-right search: images and their masks
% take transpose of images and masks
Im = (Im(topy:downy,:))';
ImMask = (ImMask(topy:downy,:))';
RefI = (RefI(topy:downy,:))';
RefIMask = (RefIMask(topy:downy,:))';

% start of the horizontal scan and end of the horizontal scan
startj=1;
endj=floor(min(STATS.BoundingBox(1) + STATS.BoundingBox(3)-midx+1, midx - STATS.BoundingBox(1)+1));

% Computing the Bhattacharya coefficient-based score function
BC_diff_LR = score(Im,RefI,ImMask,RefIMask,startj,endj,fact);
horz_scale = 30; % scale for finding maxima and minima of the vertical score function
[leftx1, rightx1]= find_largest_decreasing_segment(BC_diff_LR,horz_scale);

leftx  = leftx1(1);
rightx = rightx1(1);
leftx2 = leftx1(1);
rightx2 = rightx1(1);
leftx = leftx + midx + startj-1;
rightx = rightx + midx+ startj-1;
m_right = mean2(I(topy:downy,leftx:rightx)); % right side of line of symmetry
m_left  = mean2(I(topy:downy,2* midx - rightx:2* midx - leftx));
isleft = 0;
if m_left>m_right,
    leftx1 = 2* midx - rightx;
    rightx1 = 2* midx - leftx;
    leftx = leftx1;
    rightx = rightx1;
    isleft = 1;
end
if isleft == 1,
    figure(1), subplot(2,2,4),plot(midx - endj:midx - startj,-BC_diff_LR(end:-1:1),'r');
    subplot(2,2,4), hold on; plot(rightx,-BC_diff_LR(leftx2),'y.',leftx,-BC_diff_LR(rightx2),'c.');
else
    figure(1), subplot(2,2,4),plot(midx+startj:midx+endj,BC_diff_LR,'r');
    subplot(2,2,4), hold on; plot(leftx,BC_diff_LR(leftx2),'c.',rightx,BC_diff_LR(rightx2),'y.');
end
title('Score plot for horizontal direction');
set(gcf, 'color', [1 1 1]);

figure(1),subplot(2,2,1), hold on;
plot([leftx rightx],[topy, topy],'r');
plot([leftx rightx],[downy, downy],'g');
plot([leftx, leftx],[topy downy],'c');
plot([rightx, rightx],[topy downy],'y');

function [M,I]=skull_detect(I)

I(1:end,1)=0; I(1:end,end)=0; I(1,1:end)=0; I(end,1:end)=0;
J = imfill(I,'holes');

K = im2bw(J/max(J(:)), 0.3* graythresh(J/max(J(:))));

[L,N] = bwlabel(K);
maxa = 0; maxi=0;
for i=1:N,
    a = sum(sum(L==i));
    if a>maxa,
        maxa=a;
        maxi=i;
    end
end
L = double((L==maxi));
figure,imagesc(L),colormap(gray);axis image; drawnow;

STATS = regionprops(L,'all');
STATS.Centroid;
x0 = round(STATS.Centroid(1));
y0 = round(STATS.Centroid(2));

[h,w] = size(I);
temp = I(y0-min(y0,h-y0)+1:y0+min(y0,h-y0),x0-min(x0,w-x0)+1:x0+min(x0,w-x0));
clear I;
I = temp;
clear temp;
temp = L(y0-min(y0,h-y0)+1:y0+min(y0,h-y0),x0-min(x0,w-x0)+1:x0+min(x0,w-x0));
L = temp;
clear temp;

STATS.Orientation;
if STATS.Orientation<0,
    M = imrotate(L,-90-STATS.Orientation);
    I = imrotate(I,-90-STATS.Orientation);
else
    M = imrotate(L,90-STATS.Orientation);
    I = imrotate(I,90-STATS.Orientation);
end
close all;

function BC_diff_TD = score(Im,RefI,ImMask,RefIMask,starti,endi,fact)

BC_diff_TD  = zeros(endi-starti+1,1);

minval = max(min(Im(:)),min(RefI(:)));
maxval = min(max(Im(:)),max(RefI(:)));
offset=15;
xbins = (minval:fact:maxval);
for i = starti:endi,
    
    Tmp = Im(1:i,:);
    H_leftTop = hist(Tmp(ImMask(1:i,:)),xbins);
    clear Tmp;
    
    Tmp = RefI(1:i,:);
    H_rightTop = hist(Tmp(RefIMask(1:i,:)),xbins);
    clear Tmp;

    Tmp = Im(i:end,:);
    H_leftBottom = hist(Tmp(ImMask(i:end,:)),xbins);
    clear Tmp;

    Tmp = RefI(i:end,:);
    H_rightBottom = hist(Tmp(RefIMask(i:end,:)),xbins);
    clear Tmp;

    % normalize the histograms
    H_leftTop = H_leftTop / (sum(H_leftTop)+eps);
    H_rightTop = H_rightTop / (sum(H_rightTop)+eps);
    H_leftBottom = H_leftBottom / (sum(H_leftBottom)+eps);
    H_rightBottom = H_rightBottom / (sum(H_rightBottom)+eps);

    % compute BCs
    BC_Top = sum(sqrt(H_leftTop .* H_rightTop));
    BC_Bottom = sum(sqrt(H_leftBottom .* H_rightBottom));

    % compute difference of BCs
    if i<=starti+offset,
        BC_diff_TD(i-starti+1) = -BC_Bottom;
        if i==starti+offset,
            BC_diff_TD(1:i-starti+1) = BC_diff_TD(1:i-starti+1) + BC_Top;
        end
    elseif i>=endi-offset,
        if i==endi-offset,
            to_subs = BC_Bottom;
        end
        BC_diff_TD(i-starti+1) = BC_Top-to_subs;
    else
        BC_diff_TD(i-starti+1) = BC_Top-BC_Bottom;
    end
end


function [from, to]= find_largest_decreasing_segment(score,scale)

% first find the regional minima and maxima
hf_scale=round(scale/2);

ext_score = [ones(hf_scale,1)*score(1); score(:); ones(hf_scale,1)*score(end)];
N = length(score);
reg_minmax = zeros(N,1); 

for n=1:N,
    if min(ext_score(n:n+2*hf_scale))==score(n),
        reg_minmax(n)=-1;
    elseif max(ext_score(n:n+2*hf_scale))==score(n),
        reg_minmax(n)=1;
    end
end

% now find out largest decreasing segment
% the criterion is area rather than length
n=1;
count = 0;
while n <N-1;
    while reg_minmax(n)<1 && n<N-1,
        n = n + 1;
    end
    m=n;
    n = n+1;
    while reg_minmax(n)==0 && n<N,
        n=n+1;
    end
    if reg_minmax(n)==-1
        count = count + 1;
        thisarea(count) = 0.5*(score(m)-score(n))*(n-m);
        from(count)=m;
        to(count)=n;
    end
end
[thisarea,ind] = sort(thisarea,'descend');
from(:) = from(ind);
to(:) = to(ind);