function [md, mdq, mdHis, mdqHis, bc]=analyseCluster(matrix, labels)

    % analyse
    nr=size(matrix,1);

    mdc=0;
    mdqc=0;

    mdHis=zeros(200,1);
    mdqHis=zeros(200,1);

    md=0;
    mdq=0;
    for i=1:nr-1
        for j=i+1:nr
        % Cosine distance

            cd=min(1,dot(matrix(i,:),matrix(j,:)));
            px=ceil((cd+1)*100);

            if labels(i)~=labels(j)
                md=md+cd;
                mdc=mdc+1;

                mdHis(px)=mdHis(px)+1;

            else
                mdq=mdq+cd;
                mdqc=mdqc+1;
                
                mdqHis(px)=mdqHis(px)+1;
            end
        end
    end

   

    md=md/mdc;
    mdq=mdq/mdqc;

    mdHis=mdHis/mdc;
    mdqHis=mdqHis/mdqc;

    % Bhattacharyya Coefficient
    bc=0;
    for i=1:length(mdHis)
        bc=bc+sqrt(mdHis(i)*mdqHis(i));
    end

    if 0

        % analyse
    cor=0;
    nr=size(matrix,1);
    ds=zeros(nr,1);
    mx=zeros(nr,1);

    if nargout>4
    sm=zeros(nr,nr);
    end

    lbl=length(unique(labels));
    mdHis=zeros(200,1);
    mdqHis=zeros(200,1);

    md=0;
    mdq=0;
    nrq=0;
    for i=1:nr
        % Cosine distance
        if nargout>4
            for j=1:nr
                ds(j)=dot(matrix(i,:),matrix(j,:));
                sm(i,j)=ds(j);
            end
        else
            for j=1:nr
                ds(j)=dot(matrix(i,:),matrix(j,:));
            end
        end
       
        q=find(labels(i)==labels);
        lq=length(q);
        q(find(q==i))=[]; % remove itself!

        p=ds(find(labels(i)~=labels));


        r=ds(q);
        p(find(p>1))=1.0; % in case doublets exist
        r(find(r>1))=1.0; % in case doublets exist

        if max(r)>max(p)
            cor=cor+1;
        end


        md=md+sum(p)/(nr-lq);
        if lq>1 % Don't do anyhting if there is only one instance for this cluster
            mdq=mdq+sum(r)/(lq-1);
            nrq=nrq+1;
        end

        % Compute histograms
        px=ceil((p+1)*100);
        lpx=length(px);
        ilpx=1/lpx;
        for k=1:lpx
            mdHis(px(k))=mdHis(px(k))+ilpx;
        end

        rx=ceil((r+1)*100);
        lrx=length(rx);
        ilrx=1/lrx;

        for k=1:lrx
            mdqHis(rx(k))=mdqHis(rx(k))+ilrx;
        end


    end
    md=md/nr;
    mdq=mdq/nrq;

    mdHis=mdHis/nr;
    mdqHis=mdqHis/nr;
    end
    
end

