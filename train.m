clear all;

tic
%�����


Spk_num=10; %˵���˸���
Tra_num=10;  %ÿ��˵��������ѵ����������Ŀ

ncentres=2; %��ϳɷ���Ŀ
fs=16000; %����Ƶ��

% -- ѵ�� ---
load train.mat; 
for spk_cyc=1:Spk_num    % ����˵����
  fprintf('Train the %ith speaker\n',spk_cyc);
  tag1=1;tag2=1; %���ڻ��ܴ洢mfcc
  for sph_cyc=1:Tra_num  % ��������
     speech = tdata1{spk_cyc}{sph_cyc}; 
      %---Ԥ����,������ȡ--

     pre_sph=filter([1 -0.97],1,speech); % pre-emphasis
     win_type='M'; %������
     cof_num=20; %Number of cepstrum coefficients
     frm_len=fs*0.02; %֡����20ms
     fil_num=20; %Number of filter banks
     frm_off=fs*0.01; %֡�ƣ�10ms
     c=melcepst(pre_sph,fs,win_type,cof_num,fil_num,frm_len,frm_off); % mfcc������ȡ
     cc=c(:,1:end-1)';
     tag2=tag1+size(cc,2);
     cof(:,tag1:tag2-1)=cc;
     tag1=tag2;
  end
   
  %--- Train GMM model---
  kiter=5; %Kmeans������������
  emiter=30; %EM�㷨������������
  mix=gmm_init(ncentres,cof',kiter,'full'); % GMM�ĳ�ʼ��

  [mix,post,errlog]=gmm_em(mix,cof',emiter); % GMM�Ĳ�������
  speaker{spk_cyc}.pai=mix.priors;
  speaker{spk_cyc}.mu=mix.centres;
  speaker{spk_cyc}.sigma=mix.covars;

  clear cof mix;
end
fprintf('Training is complete! \n',spk_cyc);
save speaker.mat speaker;
toc
disp(['Time: ',num2str(toc)]);









