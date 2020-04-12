# 天池智能运维算法大赛-fengyang95 代码说明
## 解决方案
1. 利用pyspark处理原始log数据，打标签
2. 特征包括 原始特征+窗口特征+组合特征
3. 利用self-paced ensemble+lightgbm 训练和测试

## 代码运行说明
1. 运行 model文件夹内train_pandas_cls_window_7_spe_66_lgb_F37.py 处理数据和训练模型（原始数据解压后放在data/round1_train/logs文件夹）
2. 运行 code文件夹内main.py文件预测结果（线上分数最高的那个模型路径为user_data/model_data/spelgb_window7_66_leaves50_F37）