function [acc, precision, recal, iou, fmeasure] = computeStats(seg_im, gt_im)

 acc = numel(union(find(seg_im==1 & gt_im==1),find(seg_im==0 & gt_im==0)))/(prod(size(seg_im))-numel(find(gt_im==255)));

 iou = numel(intersect(find(gt_im==1),find(seg_im==1)))/numel(union(find(gt_im==1),find(seg_im==1)));

 precision = numel(intersect(find(seg_im==1),find(gt_im==1))) / numel(find(seg_im==1));

 recal = numel(intersect(find(seg_im==1),find(gt_im==1))) / numel(find(gt_im==1));

 fmeasure = 2*precision*recal/(precision + recal); 
end