function hist = get_histogram(centers, im, cl)
im = fitimage(im);
[~, desc]=vl_phow(im2single(im),'Color', cl) ;
desc = single(desc);
hist = zeros(size(centers,2),1);
for i=1:size(desc,2)
    [~, k] = min(vl_alldist(desc(:,i), centers)) ;
    hist(k) = hist(k)+1;
end
end

function im = fitimage(im)
im = im2single(im) ;
if size(im,1) > 128
    im = imresize(im, [128 NaN]) ;
end
end

