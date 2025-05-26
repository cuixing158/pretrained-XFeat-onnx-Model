function [matchedPoints1,matchedPoints2] = detectAndMatchPoints(img1,img2,featureType)
arguments
    img1
    img2
    featureType (1,1) string {mustBeMember(featureType,["orb","sift","harris"])}="orb"
end

gray1 = im2gray(img1);
gray2 = im2gray(img2);

if featureType == "orb"
    points1 = detectORBFeatures(gray1,"ScaleFactor",1.2,"NumLevels",8);
    points2 = detectORBFeatures(gray2,"ScaleFactor",1.2,"NumLevels",8);
elseif featureType=="sift"
    points1 = detectSIFTFeatures(gray1);
    points2 = detectSIFTFeatures(gray2);
elseif featureType=="harris"
    points1 = detectHarrisFeatures(gray1);
    points2 = detectHarrisFeatures(gray2);
end

[features1,validPoints1] = extractFeatures(gray1,points1);
[features2,validPoints2] = extractFeatures(gray1,points2);

indexPairs = matchFeatures(features1,features2);
matchedPoints1 = validPoints1(indexPairs(:,1),:);
matchedPoints2 = validPoints2(indexPairs(:,2),:);

% figure; 
% h = showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2,"montage");
% h.Parent.Position = [0,0,1,1];
end