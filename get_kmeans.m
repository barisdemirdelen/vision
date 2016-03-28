
function [centers] = get_kmeans(all_images, cl, numdescr, vocabsize, samplesize, load_from_file)
if load_from_file
    load(strcat('models/centers',cl, num2str(samplesize),'-', num2str(vocabsize),'.mat'));
else
    [descs] = get_descriptors(all_images,numdescr, cl);
    disp('Clustering data')
    sample_descs = single(descs);
    [centers, ~] = vl_kmeans(sample_descs, vocabsize,'distance', 'l2', 'algorithm', 'ANN');
    
    save(strcat('models/centers',cl, num2str(samplesize),'-', num2str(vocabsize),'.mat'),'centers');
end
end
