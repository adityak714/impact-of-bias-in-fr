% Similarity Distribution Plot
% Anders Hast 3/9-2024
%

function []  = simDistPlot(type, mdHis, mdqHis, md, mdq)

    %%%figure; clf
    
    if type==0
        H=bar([mdqHis mdHis ],2.4)

        H(2).FaceAlpha=0.8;
        H(1).FaceAlpha=0.8;

 
      else
        smoothedCounts = smoothdata(mdHis, 'gaussian', 5);
        smoothedCountsq = smoothdata(mdqHis, 'gaussian', 5);
        hold on;
        plot([60:200], smoothedCountsq(60:200),'LineWidth', 2); 
        plot([60:200], smoothedCounts(60:200),'LineWidth', 2); 
    end

   
   
   if nargin==5

        hold on;
       ts= textscatter(100*(1+mdq)+20,mdqHis(ceil(100*(1+mdq)))+.002,sprintf("Mean Intra-sim. %1.4f",mdq));
       ts= textscatter(100*(1+md)+20,mdHis(ceil(100*(1+md))),sprintf("Mean Inter-sim. %1.4f",md));

   else
       
   end
legend("Intra-similarity","Inter-similarity");

    axis([60 200 0 0.07])
    ax = gca; % Get current axis
    ax.XTick = [60:10:200];
    ax.XTickLabel = [-0.4:0.1:1];


end