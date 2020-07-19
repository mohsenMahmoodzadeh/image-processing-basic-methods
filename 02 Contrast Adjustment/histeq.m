I = imread('D:\2020.2.26\ComputerEngineeringUni\6 Fundamentals Of Computer Vision\FUM 98-02\Homeworks\Images\2\Camera Man.bmp');
[R,C]=size(I);
n = R*C;
Hist=zeros(256,1);
pdf = zeros(256,1);
cum = zeros(256,1);
cdf = zeros(256,1);
out = zeros(256,1);
equlizedHist = zeros(256,1);
equlizedImg = uint8(zeros(R,C));

for r = 1:R  
    for c=1:C         
        Hist(img(r,c)+1,1)=Hist(img(r,c)+1,1)+1;
        pdf(img(r,c)+1,1) = Hist(img(r,c)+1,1)/n;
    end
end


sum = 0;

for i = 1:size(pdf)
    sum = sum + Hist(i,1);
    cum(i,1) = sum;
    cdf(i,1) = cum(i,1)/n;
    out(i,1) = round(cdf(i,1) * 255);
end

for r = 1:R
    for c = 1:C
        equlizedImg(r,c) = out(I(r,c) + 1,1);
        equlizedHist(equlizedImg(r,c)+1,1)=equlizedHist(equlizedImg(r,c)+1,1)+1;
    end
end


subplot(2,2,1);
imshow(img);
title('Camera Man');

subplot(2,2,2);
bar(Hist);
title('Camera Man Histogram');

subplot(2,2,3);
imshow(equlizedImg);
title('Equalized Image Of Camera Man ');

subplot(2,2,4);
bar(equlizedHist);
title('Equalized Histogram Of Camera Man ');




