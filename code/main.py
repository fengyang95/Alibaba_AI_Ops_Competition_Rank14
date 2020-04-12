from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType,DateType
import pandas as pd
from pyspark.sql.window import Window
import numpy as np
import os
# 替换成本地的spark路径
#os.environ['SPARK_HOME']='/home/fengyang/spark-2.4.4-bin-hadoop2.7'
#os.environ['PYTHONPATH']='/home/fengyang/spark-2.4.4-bin-hadoop2.7/python'
os.environ['JAVA_HOME']='/usr/lib/jvm/jdk1.8.0_152'

useful_columns=['serial_number', 'manufacturer', 'model', 'dt',
                    'smart_12raw', 'smart_187_normalized', 'smart_187raw',
                    'smart_188_normalized', 'smart_188raw', 'smart_190_normalized', 'smart_190raw',
                    'smart_191_normalized', 'smart_191raw', 'smart_192raw', 'smart_193raw', 'smart_194_normalized',
                    'smart_194raw', 'smart_195_normalized', 'smart_195raw', 'smart_197_normalized',
                    'smart_197raw', 'smart_198_normalized', 'smart_198raw', 'smart_199raw', 'smart_1_normalized',
                    'smart_3_normalized', 'smart_4raw', 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized',
                    'smart_7raw', 'smart_9_normalized', 'smart_9raw'
                   ]


def str2datestr(s):
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    return '-'.join((year, month, day))

def preprocessing(spark_df):
    smart_feature_columns=[column for column in spark_df.columns if 'smart' in column]


    window_spec_7 = Window.partitionBy('model', 'serial_number').orderBy(
        F.datediff(F.col('dt'), F.lit('2017-07-01'))).rangeBetween(-7, 0)
    prefix_window7='window_7_'
    for smart_col in smart_feature_columns:
        spark_df=spark_df.withColumn(smart_col,F.col(smart_col).cast(DoubleType()))
        if smart_col in ['smart_1_normalized','smart_5raw','smart_7_normalized','smart_194raw','smart_199raw',
                         'smart_190raw','smart_191raw','smart_193raw','smart_195_normalized','smart_195raw']:
            spark_df = spark_df.withColumn(prefix_window7 + 'range_' + smart_col,
                                         F.max(F.col(smart_col)).over(window_spec_7) - F.min(F.col(smart_col)).over(
                                             window_spec_7))
            spark_df = spark_df.withColumn(prefix_window7 + 'std_' + smart_col,
                                         F.stddev(F.col(smart_col)).over(window_spec_7))
        #if smart_col in ['smart_187raw','smart_188raw','smart_197raw','smart_198raw']:
        #    spark_df=spark_df.withColumn(smart_col,F.when(F.col(smart_col)>0,1).otherwise(0))
        #if smart_col in ['smart_187_normalized','smart_188_normalized','smart_197_normalized','smart_198_normalized']:
        #    spark_df=spark_df.withColumn(smart_col,F.when(F.col(smart_col)<100,1).otherwise(0))
        if smart_col in ['smart_4raw','smart_5raw','smart_191raw',
                         'smart_187raw','smart_197raw','smart_198raw',
                         'smart_199raw','window_7_range_smart_199raw']:
            spark_df=spark_df.withColumn(smart_col,F.log2(F.col(smart_col)+F.lit(1.)))

    spark_df=spark_df.withColumn('smart_199raw',F.col('smart_199raw')*F.col('window_7_range_smart_199raw'))

    spark_df = spark_df.withColumn('anomaly_sum',
                                   F.col('smart_4raw') / 12 + F.col('smart_5raw') / 16  + F.col('smart_191raw') / 18
                                    + F.col('smart_198raw')/18 +F.col('smart_197raw')/18+F.col('smart_187raw')/15)

    return spark_df



if __name__ == '__main__':

    spark = SparkSession.builder.config('spark.debug.maxToStringFields', 2000)\
        .config('spark.memory.fraction',0.6)\
        .config('spark.executor.memory', '15g') \
        .config("spark.driver.memory",'15g')\
        .getOrCreate()
        #.config('spark.driver.memory', '30g') \


    tc_dir='../tcdata/disk_sample_smart_log_round2'
    testdf=None
    for file_name in os.listdir(tc_dir):
        date=file_name.split('_')[4]
        if date<'20180823':
            continue
        curr_df=spark.read.csv(os.path.join(tc_dir,file_name),header=True).select(useful_columns)
        if testdf is None:
            testdf=curr_df
        else:
            testdf=testdf.union(curr_df)

    logs_df = testdf

    logs_df = logs_df.withColumn('dt', F.udf(str2datestr)(F.col('dt')).cast(DateType()))
    logs_df=preprocessing(logs_df)
    logs_df = logs_df.withColumn('smart_4raw', F.col('smart_4raw') / 12)
    logs_df = logs_df.withColumn('smart_5raw', F.col('smart_5raw') / 16)
    logs_df = logs_df.withColumn('smart_191raw', F.col('smart_191raw') / 18)
    logs_df = logs_df.withColumn('smart_198raw', F.col('smart_198raw') / 18)
    logs_df = logs_df.withColumn('smart_197raw', F.col('smart_197raw') / 18)
    logs_df = logs_df.withColumn('smart_187raw', F.col('smart_187raw') / 15)
    cross_columns = ['smart_4raw', 'smart_5raw', 'smart_187raw', 'smart_191raw', 'smart_197raw', 'smart_198raw']
    for i in range(len(cross_columns)):
        for j in range(i + 1, len(cross_columns)):
            col_1 = cross_columns[i]
            col_2 = cross_columns[j]
            # data_df = data_df.withColumn('mul_{}_{}'.format(col_1, col_2), F.col(col_1) * F.col(col_2))
            logs_df = logs_df.withColumn('sum_{}_{}'.format(col_1, col_2), F.col(col_1) + F.col(col_2))
            """
            for k in range(j+1,len(cross_columns)):
                col_3=cross_columns[k]
                data_df=data_df.withColumn('sum_{}_{}_{}'.format(col_1,col_2,col_3),
                                           F.col(col_1)+F.col(col_2)+F.col(col_3))
            """
    #logs_df.filter(F.col('dt')=='2018-08-31').select(['window_7_std_smart_190raw', 'window_7_range_smart_191raw',
    #   'window_7_std_smart_191raw', 'window_7_range_smart_193raw',
    #   'window_7_std_smart_193raw', 'window_7_range_smart_194raw']).show()

    logs_df=logs_df.filter(F.col('dt')>='2018-09-01')
    #logs_df.select(['diff_'+column for column in useful_smart_raw_features]).show(20


    begin_date='2018-09-01'
    high_thresh_rf=0.715
    low_thresh_rf=0.715

    logs_df=logs_df.withColumn('thresh', F.lit(high_thresh_rf) - F.datediff(F.col('dt'), F.lit(begin_date)) * (high_thresh_rf - low_thresh_rf) / 30)

    testa_pd_all = logs_df.toPandas()

    feature_columns=['smart_12raw', 'smart_187_normalized', 'smart_187raw',
       'smart_188_normalized', 'smart_188raw', 'smart_190_normalized',
       'smart_190raw', 'smart_191_normalized', 'smart_191raw', 'smart_192raw',
       'smart_193raw', 'smart_194_normalized', 'smart_194raw',
       'smart_195_normalized', 'smart_195raw', 'smart_197_normalized',
       'smart_197raw', 'smart_198_normalized', 'smart_198raw', 'smart_199raw',
       'smart_1_normalized', 'smart_3_normalized', 'smart_4raw',
       'smart_5_normalized', 'smart_5raw', 'smart_7_normalized', 'smart_7raw',
       'smart_9_normalized', 'smart_9raw', 'window_7_range_smart_190raw',
       'window_7_std_smart_190raw', 'window_7_range_smart_191raw',
       'window_7_std_smart_191raw', 'window_7_range_smart_193raw',
       'window_7_std_smart_193raw', 'window_7_range_smart_194raw',
       'window_7_std_smart_194raw', 'window_7_range_smart_195_normalized',
       'window_7_std_smart_195_normalized', 'window_7_range_smart_195raw',
       'window_7_std_smart_195raw', 'window_7_range_smart_199raw',
       'window_7_std_smart_199raw', 'window_7_range_smart_1_normalized',
       'window_7_std_smart_1_normalized', 'window_7_range_smart_5raw',
       'window_7_std_smart_5raw', 'window_7_range_smart_7_normalized',
       'window_7_std_smart_7_normalized', 'anomaly_sum',
       'sum_smart_4raw_smart_5raw', 'sum_smart_4raw_smart_187raw',
       'sum_smart_4raw_smart_191raw', 'sum_smart_4raw_smart_197raw',
       'sum_smart_4raw_smart_198raw', 'sum_smart_5raw_smart_187raw',
       'sum_smart_5raw_smart_191raw', 'sum_smart_5raw_smart_197raw',
       'sum_smart_5raw_smart_198raw', 'sum_smart_187raw_smart_191raw',
       'sum_smart_187raw_smart_197raw', 'sum_smart_187raw_smart_198raw',
       'sum_smart_191raw_smart_197raw', 'sum_smart_191raw_smart_198raw',
       'sum_smart_197raw_smart_198raw', 'model']
    print(len(feature_columns))


    test_X = testa_pd_all[feature_columns]
    test_X['model'] = test_X['model'].astype(int) - 1

    #test_X = pd.DataFrame(imp.fit_transform(test_X), columns=test_X.columns)

    test_ans = testa_pd_all[['serial_number', 'manufacturer', 'model', 'dt','thresh']]

    import joblib


    model_lgb_dir = '../user_data/model_data/spelgb_window7_66_leaves50_F37'
    models_lgb = []
    model_names = os.listdir(model_lgb_dir)
    for model_name in model_names:
        models_lgb.append(joblib.load(os.path.join(model_lgb_dir, model_name)))


    def predict_proba(X, models):
        y_pred = np.array(
            [model.predict_proba(X) for model in models]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        return y_pred


    prediction_lgb = predict_proba(test_X, models_lgb)[:, 1]

    prediction = pd.DataFrame(np.c_[prediction_lgb], columns=['prediction_lgb',
                                                              ])

    test_ans = pd.concat((test_ans, prediction), axis=1)
    # testa_df_all=testa_df_all.withColumn('preds_'+str(index),F.when(testa_df_all['preds_'+str(index)]>threshes[index],1).otherwise(0))

    #test_ans.hist(column='prediction',bins=100)
    #plt.show()
    import matplotlib.pyplot as plt

    test_a_preds = test_ans[test_ans['prediction_lgb'] > test_ans['thresh']]

    #print(test_a_preds['prediction'])
    #print(test_a_preds.columns)
    test_a_preds=test_a_preds[['manufacturer', 'model', 'serial_number', 'dt']]

    from pyspark.sql.types import StructType,StructField
    schema=StructType([
        StructField('manufacturer',StringType(),True),
        StructField('model',StringType(),True),
        StructField('serial_number',StringType(),True),
        StructField('dt',DateType(),False),

    ])
    submit = spark.createDataFrame(test_a_preds,schema).repartition(1)

    submit.registerTempTable('submit')
    #print('write result')
    submit_mindt = spark.sql("select manufacturer,model,serial_number,min(dt) "
                             "from submit group by manufacturer, model, serial_number "
                             "order by min(dt) asc,serial_number asc"
                             )
    print('submit:',submit_mindt.count())
    submit_mindt.toPandas().to_csv("../result.csv", header=False, index=None)


