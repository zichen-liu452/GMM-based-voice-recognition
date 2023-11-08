% -- Recongition ---
clear all;
load test.mat;
load speaker.mat;   % Load the trained model
Spk_num=3; 
Tes_num=1;  %Number of voices to be recognized for each speaker
fs=16000; %Sample Frequency
ncentres=2; %Number of mixed ingredients

for spk_cyc=1:Spk_num    % Traverse speakers
  for sph_cyc=1:Tes_num  % Traverse speech
     fprintf('Start recognizing the %ith voice of the %ith speaker\n',spk_cyc,sph_cyc); 
     speech = rdata1{spk_cyc}{sph_cyc};


     %--- pre-processing, feature extraction --
     pre_sph=filter([1 -0.97],1,speech);
     win_type='M'; %Hamming window
     cof_num=20; %Number of cepstrum coefficients
     frm_len=fs*0.02; %Frame length£º20ms
     fil_num=20; %Number of filter banks
     frm_off=fs*0.01; %frame shift£º10ms
     c=melcepst(pre_sph,fs,win_type,cof_num,fil_num,frm_len,frm_off); %(Frames)*(cof_num)
     cof=c(:,1:end-1); %N*D
     
     %----Recongitio---
     MLval=zeros(size(cof,1),Spk_num);
     for b=1:Spk_num 
     pai=speaker{b}.pai;
     for k=1:ncentres 
       mu=speaker{b}.mu(k,:);
       sigma=speaker{b}.sigma(:,:,k);
       pdf=mvnpdf(cof,mu,sigma);
       MLval(:,b)=MLval(:,b)+pdf*pai(k); 
     end
    end
    logMLval=log((MLval)+eps);
    sumlog=sum(logMLval,1);
    [maxsl,idx]=max(sumlog); 
    fprintf('Recognition result: %ith speaker\n',idx);     
     
  end
end