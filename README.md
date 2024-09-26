
## 執行環境
window 11
python 3.9.19
torch 2.1.1
## 安裝套件
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

## Dataset 說明
cocochorales
https://magenta.tensorflow.org/datasets/cocochorales#download
maestro
https://magenta.tensorflow.org/datasets/maestro#v300
GuitarSet
https://guitarset.weebly.com/

IMSLP Petrucci 音樂圖書館 B小調第一小提琴組曲BWV 1002 
https://imslp.org/wiki/Violin_Partita_No.1_in_B_minor%2C_BWV_1002_(Bach%2C_Johann_Sebastian)

訓練用資料集
```
// violin
// |-- violin_5
//    |-- audio
//        |-- xxxx.wav
//        |-- ...
//    |-- loudness
//    |-- pitch
// |-- violin_48     
// |-- violin_48_effect
```

測試用資料集
```
// violin_test
// |-- guitar_6
//    |-- audio
//        |-- xxxx.wav
//        |-- ...
//    |-- loudness
//    |-- pitch
// |-- pinao_48   
// |-- voice_8   
```  

## violin_subset
// 運行程式建立violin_subset資料夾，獲取test、train、valid資料夾下的檔案列表，之後建立keys_xxx.txt紀錄
python violin_subset.py
```
// violin_subset
//    |-- keys_test.tx
//    |-- keys_train.tx   
//    |-- keys_valid.tx  
```
## RMVPE 提取器說明
`-w` 需要計算的音檔位置 `-p` RMEVP 模型放置位置  `-t` 指定哪個GPU計算，0為CPU計算
執行範例
```
cd  so-vits-svc-rmvpe
python .\prepare\preprocess_rmvpe.py -w D:\wenzo\wave_synth_2024_08_14\so-vits-svc-rmvpe\test\ -p D:\wenzo\wave_synth_2024_08_14\so-vits-svc-rmvpe\pretrain\ -t 1
```
生成的計算結果會在`.\pretrain\audio`資料夾中
//so-vits-svc-rmvpe
//  |-- pretrain
//      |-- audio   
//          |-- xxxxxxx_pitch.npy  
RMVPE模型下載
(https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt)

## Train and Test
更改 config_violin.yaml來改變test、train 時的參數
```
## 預處理指令
python preprocess.py
##訓練指令
python train.py
##測試指令
python test.py
```
`train_local.py` 驗證 train.py是否可以順利執行

## 執行順序
    1.先執行 python violin_subset.py 建立檔案列表txt
    2.進行資料預處理 python preprocess.py
    3.調整進行訓練的config_violin.yaml
    4.執行 python train.py 進行訓練
    5.最後 python test.py 根據指定訓練模型生出音檔

## other
`composite_function.py` 裡面有對音訊處理的各種功能
`wlpc.py` `wlpc_test.py`wlpc濾波器試作品 
`server.py` python Flask API 套件執行程式碼，用於docker-compose 對外接口建立
`docker-compose.yaml` 可用docker 啟用訓練環境
`Dockerfile` 可以用於建立自己的docker images，調整完執行  `docker build .` 指令建立images
