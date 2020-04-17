# [天池智能运维算法大赛](https://tianchi.aliyun.com/competition/entrance/231775/introduction)-fengyang95-Rank14
[天池智能运维大赛2020踩坑分享](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.15.3f9a77c3U6M5tP&postId=102983)  
[答辩PPT](https://github.com/fengyang95/Alibaba_AI_Ops_Competition_Rank14/blob/master/Alibaba_AIOps_fengyang95_%E6%9D%8E%E5%85%83%E9%B9%8F.pdf)
## 解决方案
1. 利用pyspark处理原始log数据，打标签
2. 特征包括 原始特征+窗口特征+组合特征
3. 利用self-paced ensemble+lightgbm 训练和测试

## 代码运行说明
1. 运行 model文件夹内train_pandas_cls_window_7_spe_66_lgb_F37.py 处理数据和训练模型（原始数据解压后放在data/round1_train/logs文件夹）
2. 运行 code文件夹内main.py文件预测结果（线上分数最高的那个模型路径为user_data/model_data/spelgb_window7_66_leaves50_F37）
