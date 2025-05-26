function [matchedPoints1,matchedPoints2] = xfeatSemiDenseMatch(img1,img2,xfeatFile)
% Brief: xfeat半稠密特征点匹配深度学习模型推理，直接返回粗糙的匹配点集
% Details:
%    使用xfeat算法导入onnx模型进行端到端的特征点匹配，对任意输入同等大小分辨率的RGB三通道图像直接返回半稠密匹配的粗糙点集。
%
% Syntax:
%     [matchedPoints1,matchedPoints2] = xfeatSemiDenseMatch(img1,img2)
%
% Inputs:
%    img1 - [m,n,3] size,[uint8] type,第一幅待匹配的图像
%    img2 - [m,n,3] size,[uint8] type,第二幅待匹配的图像
%
% Outputs:
%    matchedPoints1 - [numPts,2] size,[double] type,对应第一幅图像匹配的点集像素坐标，每行形如[x,y].
%    matchedPoints2 - [numPts,2] size,[double] type,对应第二幅图像匹配的点集像素坐标，每行形如[x,y].
%
% Example:
%    None
%
% See also:
% https://github.com/cuixing158/Accelerated-Features/blob/main/my_onnx_export.py

% Author:                          cuixingxing
% Email:                           cuixingxing150@gmail.com
% Created:                         17-Apr-2025 16:17:30
% Version history revision notes:
%                                  None
% Implementation In Matlab R2025a
% Copyright © 2025 TheMatrix.All Rights Reserved.
%

arguments
    img1 (:,:,3) uint8
    img2 (:,:,3) uint8 {mustBeSameSize(img1,img2)}
    xfeatFile (1,:) char = "params.mat"
end

persistent params
if isempty(params)
    s = load(xfeatFile);
    params = s.params;
end

if canUseGPU()
    img1 = gpuArray(img1);
    img2 = gpuArray(img2);
end

[mkpts1,mkpts2] = xfeatFcn(img1,img2,params,OutputDataPermutation='none');

if canUseGPU()
    mkpts1 = gather(mkpts1);
    mkpts2 = gather(mkpts2);
end

matchedPoints1 = double(mkpts1);
matchedPoints2 = double(mkpts2);
end

function mustBeSameSize(a, b)
if ~isequal(size(a), size(b))
    error('The sizes of the inputs must be equal.');
end
end