function descs = get_descriptors(arrImages, numdescr, cl)
descs = [];
for i=1:size(arrImages, 2)
    
    im = fitimage(arrImages{i});
    [~, d]=vl_phow(im2single(im),'Color', cl) ;
    k = randperm(size(d,2));
    sample_descs = d(:,k(1:numdescr));
    descs = horzcat(descs , single(sample_descs));
    disp(i)
    
end
end

function im = fitimage(im)
im = im2single(im) ;
if size(im,1) > 128
    im = imresize(im, [128 NaN]) ;
end
end


