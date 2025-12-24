% Loading the labels and matrices of the embeddings
load("/MATLAB Drive/labels.mat")
load("/MATLAB Drive/matrix.mat")

[md, mdq, mdhis, mdqhis, bc] = analyseCluster(matrix, labels)

simDistPlot(0, mdhis, mdqhis, md, mdq) % For creating with the histogram bins
simDistPlot(1, mdhis, mdqhis, md, mdq) % For creating without the histogram bins