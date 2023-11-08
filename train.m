clear all;

tic
%代码块


Spk_num=10; %说话人个数
Tra_num=10;  %每个说话人用于训练的语音数目

ncentres=2; %混合成分数目
fs=16000; %采样频率

% -- 训练 ---
load train.mat; 
for spk_cyc=1:Spk_num    % 遍历说话人
  fprintf('Train the %ith speaker\n',spk_cyc);
  tag1=1;tag2=1; %用于汇总存储mfcc
  for sph_cyc=1:Tra_num  % 遍历语音
     speech = tdata1{spk_cyc}{sph_cyc}; 
      %---预处理,特征提取--

     pre_sph=filter([1 -0.97],1,speech); % pre-emphasis
     win_type='M'; %汉明窗
     cof_num=20; %Number of cepstrum coefficients
     frm_len=fs*0.02; %帧长：20ms
     fil_num=20; %Number of filter banks
     frm_off=fs*0.01; %帧移：10ms
     c=melcepst(pre_sph,fs,win_type,cof_num,fil_num,frm_len,frm_off); % mfcc特征提取
     cc=c(:,1:end-1)';
     tag2=tag1+size(cc,2);
     cof(:,tag1:tag2-1)=cc;
     tag1=tag2;
  end
   
  %--- Train GMM model---
  kiter=5; %Kmeans的最大迭代次数
  emiter=30; %EM算法的最大迭代次数
  mix=gmm_init(ncentres,cof',kiter,'full'); % GMM的初始化

  [mix,post,errlog]=gmm_em(mix,cof',emiter); % GMM的参数估计
  speaker{spk_cyc}.pai=mix.priors;
  speaker{spk_cyc}.mu=mix.centres;
  speaker{spk_cyc}.sigma=mix.covars;

  clear cof mix;
end
fprintf('Training is complete! \n',spk_cyc);
save speaker.mat speaker;
toc
disp(['Time: ',num2str(toc)]);









