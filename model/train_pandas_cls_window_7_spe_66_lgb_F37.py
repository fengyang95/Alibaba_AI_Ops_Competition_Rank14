from pyspark.sql.types import DateType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from sklearn.metrics import precision_recall_curve,auc
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import sklearn
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
import math
import os
import joblib
import pandas as pd
# spark home
os.environ['SPARK_HOME']='/home/fengyang/spark-2.4.4-bin-hadoop2.7'


from pyspark.sql import Window

# modified from https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/self_paced_ensemble.py
class SelfPacedEnsemble(ClassifierMixin):
    """ Self-paced Ensemble (SPE)

    Parameters
    ----------

    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
        The base estimator to fit on self-paced under-sampled subsets of the dataset.
        NO need to support sample weighting.
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.

    hardness_func :  function, optional
        (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)
        User-specified classification hardness function
            Parameters:
                y_true: 1-d array-like, shape = [n_samples]
                y_pred: 1-d array-like, shape = [n_samples]
            Returns:
                hardness: 1-d array-like, shape = [n_samples]

    n_estimators :  integer, optional (default=10)
        The number of base estimators in the ensemble.

    k_bins :        integer, optional (default=10)
        The number of hardness bins that were used to approximate hardness distribution.

    random_state :  integer / RandomState instance / None, optional (default=None)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        `numpy.random`.


    Attributes
    ----------

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    Example:
    ```
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from self_paced_ensemble import SelfPacedEnsemble
    from utils import make_binary_classification_target, imbalance_train_test_split

    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, True)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
            X, y, test_size=0.2, random_state=42)

    def absolute_error(y_true, y_pred):
        # Self-defined classification hardness function
        return np.absolute(y_true - y_pred)

    spe_window7_t0.68_P29_R34_F31.4 = SelfPacedEnsemble(
        base_estimator=DecisionTreeClassifier(),
        hardness_func=absolute_error,
        n_estimators=10,
        k_bins=10,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )
    print('auc_prc_score: {}'.format(spe_window7_t0.68_P29_R34_F31.4.score(X_test, y_test)))
    ```

    """

    def __init__(self,
                 base_estimator=LGBMClassifier(),
                 hardness_func=lambda y_true, y_pred: np.absolute(y_true - y_pred),
                 n_estimators=10,
                 k_bins=10,
                 random_state=None,
                 feature_columns=None,
                 label_binarier=False):
        super().__init__()
        self.base_estimator_ = base_estimator
        self.estimators_ = []
        self._hardness_func = hardness_func
        self._n_estimators = n_estimators
        self._k_bins = k_bins
        self._random_state = random_state
        self._feature_columns=feature_columns
        self._label_binarier=label_binarier

    def _fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        if not isinstance(X,pd.DataFrame) and self._feature_columns is not None and self._label_binarier is False:

            X=pd.DataFrame(X,index=None,columns=self._feature_columns,dtype=np.float)

            #X=pd.DataFrame(X,columns=self._feature_columns,dtype=np.float)
            X['model']=X['model'].astype(np.int)#.astype('category')
            #print('transpose')
            if isinstance(self.base_estimator_,GBDTLRClassifier):
                return sklearn.base.clone(self.base_estimator_).fit(X, y, gbdt__categorical_feature=[65])
            else:
                return sklearn.base.clone(self.base_estimator_).fit(X, y,categorical_feature=[65])

        if self._label_binarier is True:
            return sklearn.base.clone(self.base_estimator_).fit(X,y)

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        np.random.seed(self._random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj.iloc[idx], X_min])
        y_train = np.concatenate([y_maj.iloc[idx], y_min])
        return X_train, y_train

    def _self_paced_under_sampling(self,
                                   X_maj, y_maj, X_min, y_min, i_estimator):
        """Private function used to perform self-paced under-sampling."""
        # Update hardness value estimation
        hardness = self._hardness_func(y_maj, self._y_pred_maj)

        # If hardness values are not distinguishable, perform random smapling
        if hardness.max() == hardness.min():
            X_train, y_train = self._random_under_sampling(X_maj, y_maj, X_min, y_min)
        # Else allocate majority samples into k hardness bins
        else:
            step = (hardness.max() - hardness.min()) / self._k_bins
            bins = []
            ave_contributions = []
            for i_bins in range(self._k_bins):
                idx = (
                        (hardness >= i_bins * step + hardness.min()) &
                        (hardness < (i_bins + 1) * step + hardness.min())
                )
                # Marginal samples with highest hardness value -> kth bin
                if i_bins == (self._k_bins - 1):
                    idx = idx | (hardness == hardness.max())
                bins.append(X_maj[idx])
                ave_contributions.append(hardness[idx].mean())

            # Update self-paced factor alpha
            alpha = np.tan(np.pi * 0.5 * (i_estimator / (self._n_estimators - 1)))
            # Caculate sampling weight
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            # Caculate sample number from each bin
            n_sample_bins = len(X_min) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int) + 1

            # Perform self-paced under-sampling
            sampled_bins = []
            for i_bins in range(self._k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self._random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]),
                        min(len(bins[i_bins]), n_sample_bins[i_bins]),
                        replace=False)
                    sampled_bins.append(bins[i_bins].iloc[idx])
            X_train_maj = np.concatenate(sampled_bins, axis=0)
            y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])
            X_train = np.concatenate([X_train_maj, X_min])
            y_train = np.concatenate([y_train_maj, y_min])

        return X_train, y_train

    def fit(self, X, y, label_maj=0, label_min=1):
        """Build a self-paced ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels).

        label_maj : int, bool or float, optional (default=0)
            The majority class label, default to be negative class.

        label_min : int, bool or float, optional (default=1)
            The minority class label, default to be positive class.

        Returns
        ------
        self : object
        """
        self.estimators_ = []
        if self._label_binarier is True:
            label_binarier=LabelBinarizer()
            label_binarier.fit(np.arange(500000))
            X_id=label_binarier.transform(X['model_serial'])
            X.drop(['model_serial'],axis=1,inplace=True)
            X=np.concatenate((X.to_numpy(),X_id),axis=1)
        # Initialize by spliting majority / minority set
        X_maj = X[y == label_maj]
        y_maj = y[y == label_maj]
        X_min = X[y == label_min]
        y_min = y[y == label_min]

        # Random under-sampling in the 1st round (cold start)
        X_train, y_train = self._random_under_sampling(
            X_maj, y_maj, X_min, y_min)
        self.estimators_.append(
            self._fit_base_estimator(
                X_train, y_train))
        self._y_pred_maj = self.predict_proba(X_maj)[:, 1]

        # Loop start
        for i_estimator in range(1, self._n_estimators):
            X_train, y_train = self._self_paced_under_sampling(
                X_maj, y_maj, X_min, y_min, i_estimator, )
            self.estimators_.append(
                self._fit_base_estimator(
                    X_train, y_train))
            # update predicted probability
            n_clf = len(self.estimators_)
            y_pred_maj_last_clf = self.estimators_[-1].predict_proba(X_maj)[:, 1]
            self._y_pred_maj = (self._y_pred_maj * (n_clf - 1) + y_pred_maj_last_clf) / n_clf

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        if self._label_binarier is True:
            label_binarier=LabelBinarizer()
            label_binarier.fit(np.arange(500000))
            X_id=label_binarier.transform(X['model_serial'])
            X.drop(['model_serial'],axis=1,inplace=True)
            X=np.concatenate((X.to_numpy(),X_id),axis=1)
        y_pred = np.array(
            [model.predict_proba(X) for model in self.estimators_]
        ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)
        print(y_pred.shape)
        return y_pred

    def predict(self, X):
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:, 1].reshape(1, -1), threshold=0.5)[0]
        return y_pred_binarized

    def score(self, X, y):

        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])

def get_logs(logs_dir,useful_columns):
    """
    read all logs , merge them and select useful original columns
    :param logs_dir: e.x. ../data/logs
    :return: pyspark DataFrame logs_df
    """
    logs_df_all=None
    for file_name in os.listdir(logs_dir):
        log_path=os.path.join(logs_dir,file_name)
        if log_path.endswith('.csv'):
            curr_logs_df=spark.read.csv(log_path,header=True)
            curr_logs_df=curr_logs_df.select(useful_columns)
            if logs_df_all is None:
                logs_df_all=curr_logs_df
            else:
                logs_df_all=logs_df_all.union(curr_logs_df)
    return logs_df_all


from pyod.utils import precision_n_scores
def precise_n(y,y_pred):
    precise_n_score=precision_n_scores(y,y_pred)
    return 'precise_n',precise_n_score,True

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
                                   F.col('smart_4raw')/12  + F.col('smart_5raw')/16   + F.col('smart_191raw')/18
                                    + F.col('smart_198raw')/18 +F.col('smart_197raw')/18+F.col('smart_187raw')/15)

    return spark_df



if __name__ == '__main__':
    spark = SparkSession.builder.config('spark.debug.maxToStringFields', 2000) \
        .config('spark.memory.fraction', 0.6).config('spark.executor.memory', '100g') \
        .config('spark.driver.maxResultSize','100g')\
        .config('spark.driver.memory', '60g').getOrCreate()

    useful_columns=['serial_number', 'manufacturer', 'model', 'dt',
                    'smart_12raw', 'smart_187_normalized', 'smart_187raw',
                    'smart_188_normalized', 'smart_188raw', 'smart_190_normalized', 'smart_190raw',
                    'smart_191_normalized', 'smart_191raw', 'smart_192raw', 'smart_193raw', 'smart_194_normalized',
                    'smart_194raw', 'smart_195_normalized', 'smart_195raw', 'smart_197_normalized',
                    'smart_197raw', 'smart_198_normalized', 'smart_198raw', 'smart_199raw', 'smart_1_normalized',
                    'smart_3_normalized', 'smart_4raw', 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized',
                    'smart_7raw', 'smart_9_normalized', 'smart_9raw'

                   ]

    if not os.path.exists('../data/round1_train/logs_66.csv'):
        smart_logs_dir = '../data/round1_train/logs'
        logs_df = get_logs(smart_logs_dir, useful_columns).repartitionByRange(32, 'serial_number', 'model')
        logs_df = logs_df.withColumn('dt', F.udf(str2datestr)(F.col('dt')))#.cast(DateType()))
        data_df = logs_df.select(useful_columns)
        data_df = preprocessing(data_df)
        data_df.write.csv('../data/round1_train/logs_66.csv',
                          header=True, sep=',', mode='overwrite')
    else:
        data_df=spark.read.csv('../data/round1_train/logs_66.csv',header=True)
        smart_columns=[column for column in data_df.columns if 'smart' in column]+['anomaly_sum']
        for col in smart_columns:
            data_df=data_df.withColumn(col,F.col(col).cast(DoubleType()))

        #data_df=data_df.withColumn('dt',F.col('dt').cast(DateType()))
    # feature cross

    data_df = data_df.withColumn('smart_4raw', F.col('smart_4raw') / 12)
    data_df = data_df.withColumn('smart_5raw', F.col('smart_5raw') / 16)
    data_df = data_df.withColumn('smart_191raw', F.col('smart_191raw') / 18)
    data_df = data_df.withColumn('smart_198raw', F.col('smart_198raw') / 18)
    data_df = data_df.withColumn('smart_197raw', F.col('smart_197raw') / 18)
    data_df = data_df.withColumn('smart_187raw', F.col('smart_187raw') / 15)
    cross_columns = ['smart_4raw', 'smart_5raw', 'smart_187raw', 'smart_191raw', 'smart_197raw', 'smart_198raw']
    for i in range(len(cross_columns)):
        for j in range(i + 1, len(cross_columns)):
            col_1 = cross_columns[i]
            col_2 = cross_columns[j]
            data_df = data_df.withColumn('sum_{}_{}'.format(col_1, col_2), F.col(col_1)+F.col(col_2))

            #for k in range(j+1,len(cross_columns)):
            #    col_3=cross_columns[k]
            #    data_df=data_df.withColumn('sum_{}_{}_{}'.format(col_1,col_2,col_3),
            #                               F.col(col_1)+F.col(col_2)+F.col(col_3))

    print(data_df.count())

    data_df = data_df.withColumn('month', F.udf(lambda dt: dt[:7])(F.col('dt')))

    data_df=data_df.withColumn('dt',F.col('dt').cast(DateType()))
    tag=spark.read.csv('../data/round1_train/disk_sample_fault_tag.csv',header=True).withColumn('fault_time',F.col('fault_time').cast(DateType()))
    data_df=data_df.join(tag,on=['serial_number','model','manufacturer'],how='left').select(data_df.columns+['fault_time'])


    data_df = data_df.withColumn('label', F.when(
        (F.col('fault_time').isNotNull()) & (F.datediff(F.col('fault_time'), F.col('dt')) <= 30), 1).otherwise(0))
   
    # 这是训练线上模型的日期选择，线下验证是用2017-08-1到2018-06-01数据训练，用2018-07-01到2018-08-01数据验证
    test1_begin_date = '2018-08-01'
    test1_end_date = '2018-09-01'

    train_begin_date = '2017-08-01'
    train_end_date = '2018-08-01'
    tags_end_date='2018-09-01'

    val_begin_date='2018-06-01'
    val_end_date='2018-07-01'

    train_df = data_df.filter(data_df['dt'] >= train_begin_date).filter(data_df['dt'] < train_end_date)
    test_df = data_df.filter(data_df['dt'] >= test1_begin_date).filter(data_df['dt'] < test1_end_date)#.filter(F.col('model')==1)
    val_df=data_df.filter(data_df['dt']>=val_begin_date).filter(data_df['dt']<val_end_date)
    use_only_fault_disk=False
    if use_only_fault_disk is True:

        # only fault tag
        train_df_n = train_df.filter(F.col('fault_time').isNotNull()).filter(F.col('label') == 0)
        train_df_p = data_df.filter(F.col('label') == 1).filter(F.datediff(F.col('fault_time'), F.col('dt')) <= 30) \
            .filter(F.col('dt') >= '2017-08-01').filter(F.col('fault_time') < tags_end_date)
        train_df=train_df_n.union(train_df_p)


    else:
        window = Window.partitionBy('serial_number', 'model', 'month').orderBy(F.rand(seed=2019))
        train_df_p = data_df.filter(F.col('label') == 1).filter(F.datediff(F.col('fault_time'),F.col('dt'))<=30)\
            .filter(F.col('dt')>='2017-08-01').filter(F.col('fault_time')<tags_end_date)#.withColumn('topn', F.row_number().over(window)).where(
        #F.col('topn') <= 2).select(train_df.columns)
        train_df_n= train_df.filter(F.col('label')==0).withColumn('topn', F.row_number().over(window)).where(F.col('topn') <= 2).select(train_df.columns)

        train_df_ext_n = train_df.filter(F.col('fault_time').isNotNull()).filter(F.col('label') == 0)
        train_df=train_df_n.union(train_df_p)#.union(train_df_ext_n)
        #train_df=train_df.withColumn('model_serial',F.when(F.col('fault_time').isNotNull(),F.col('model_serial')).otherwise(F.lit(-1)))



    #window_model_serial = Window.partitionBy('serial_number', 'model').orderBy(F.rand(seed=2019))
    #label_data = data_df.withColumn('topn', F.row_number().over(window_model_serial)).where(F.col('topn') <= 1).select(
    #    ['model_serial'])
    #print(label_data.count())
    #from sklearn.preprocessing import LabelEncoder
    #label_encoder=LabelEncoder()
    #label_encoder.fit(label_data.repartiton(1).toPandas())
    #joblib.dump(label_encoder,'label_encoder')
    print('train_p:',train_df.filter(F.col('label')==1).count())
    print('train_n:',train_df.filter(F.col('label')==0).count())

    feature_columns = [column for column in test_df.columns if
                       column not in ['label', 'dt', 'serial_number', 'month', 'manufacturer', 'fault_time','model']]+['model']

    test_pd = test_df.repartition(1).toPandas()

    #print(test_pd.describe())
    #test_pd.describe().to_csv('test.csv')
    print('test:', len(test_pd))
    import time
    beg=time.time()

    train_pd = train_df.repartition(1).toPandas()
    train_X = train_pd[feature_columns]
    # train_X=onehot(train_X)
    #train_X['model_serial']=label_encoder.transform(train_X['model_serial'])
    train_X['model']=train_X['model'].astype(np.int)-1
    train_y = train_pd['label'].astype(np.int)


    test_X1 = test_pd[feature_columns]
    #test_X1=onehot(test_X1)
    #test_X1['model_serial']=label_encoder.transform(test_X1['model_serial'])
    test_X1['model']=test_X1['model'].astype(np.int)-1
    test_y1 = test_pd['label'].astype(np.int)

    val_pd=val_df.repartition(1).toPandas()
    val_X=val_pd[feature_columns]
    #val_X=onehot(val_X)
    #val_X['model_serial']=label_encoder.transform(val_X['model_serial'])
    val_X['model']=val_X['model'].astype(np.int)-1
    val_y=val_pd['label'].astype(np.int)

    #joblib.dump(label_encoder,'label_encoder')


    print(train_X.columns)
    print(len(train_X.columns))


    import gc
    gc.collect()



    spe_lgb = SelfPacedEnsemble(base_estimator=LGBMClassifier(boosting_type='gbdt', num_leaves=50, n_estimators=20,
                                                              learning_rate=0.1,bagging_fraction=0.8,
                                                              feature_fraction=0.8,min_data_per_group=500,
                                                              max_cat_threshold=32,
                                                              ),
                                feature_columns=train_X.columns,
                                k_bins=40, n_estimators=20,
                                hardness_func=lambda y_pred,y_label:(y_pred-y_label)**2
                                )
    from sklearn2pmml.ensemble import GBDTLRClassifier
    from sklearn.linear_model import LogisticRegression



    spe_lgb.fit(train_X, train_y)
    dir = '../user_data/model_data/spelgb_window7_66'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i, model in enumerate(spe_lgb.estimators_):
        joblib.dump(model, os.path.join(dir, 'spe_{:02d}'.format(i)))
    gc.collect()



    test_predsy_spe_lgb = spe_lgb.predict_proba(test_X1)[:, 1]
    gc.collect()

    from collections import defaultdict

    spe_importance_dict = defaultdict(float)

    for i in range(20):
        for importance, name in zip(spe_lgb.estimators_[i].feature_importances_, train_X.columns):
            spe_importance_dict[name] += round(importance, 3)
    import json

    with open('spelgb_importance_66.json', 'w') as f:
        json.dump(spe_importance_dict, f)


    test_predsy_spe_lgb_cat=pd.DataFrame(test_predsy_spe_lgb,
                                         index=None,
                                     columns=['prediction_spe_lgb'
                                              ])

    test_preds_lgb_cat= pd.concat((test_predsy_spe_lgb_cat, test_pd[['label', 'model', 'serial_number', 'dt']]), axis=1)

    import matplotlib.pyplot as plt
    test_preds_lgb_cat.hist(column='prediction_spe_lgb', bins=100)
    plt.show()
    plot_pr = True
    if plot_pr is True:
        import matplotlib.pyplot as plt

        # spe lgb
        spe_lgb_train_pred= spe_lgb.predict_proba(train_X)[:, 1]
        precision, recall, _ = precision_recall_curve(train_y,
                                                              spe_lgb_train_pred)
        aucpr=auc(recall,precision)
        roc_auc=roc_auc_score(train_y,spe_lgb_train_pred)
        plt.plot(recall, precision, label='train_spe_lgb:{:.4f}{:.3f}'.format(aucpr,roc_auc))

        spe_lgb_val_pred = spe_lgb.predict_proba(val_X)[:, 1]
        precision, recall, _ = precision_recall_curve(val_y,
                                                      spe_lgb_val_pred)
        aucpr = auc(recall, precision)
        roc_auc = roc_auc_score(val_y, spe_lgb_val_pred)
        plt.plot(recall, precision, label='val_spe_lgb:{:.4f}{:.3f}'.format(aucpr, roc_auc))


        precision, recall, thresholds = precision_recall_curve(test_preds_lgb_cat['label'],
                                                               test_preds_lgb_cat['prediction_spe_lgb'])
        aucpr=auc(recall,precision)
        roc_auc=roc_auc_score(test_preds_lgb_cat['label'], test_preds_lgb_cat['prediction_spe_lgb'])
        plt.plot(recall, precision, label='spe_lgb:{:.4f}-{:.3f}'.format(aucpr,roc_auc))

        max_f1_score = 0
        best_thresh_spe_lgb = 0

        for p, r, \
            t in zip(precision, recall, thresholds):

            f1_score = 2 * (p * r) / (p + r)
            if f1_score > max_f1_score:
                best_thresh_spe_lgb = t
                max_f1_score = f1_score
        print('spe lgb:')
        print('best_thresh:', best_thresh_spe_lgb)
        print('max_f1_score:', max_f1_score)
        gc.collect()


        plt.title('models')
        plt.legend()
        plt.show()

        preds_lgb_cat_spark=spark.createDataFrame(test_preds_lgb_cat).repartition(1)


        def cal_F1(submit,fault_tag_df):
            npp = submit.count()
            tmp = submit.join(fault_tag_df, on=['model', 'serial_number'], how='left')
            ntpp = tmp.filter(F.col('fault_time').isNotNull()).filter(
                F.date_sub(tmp['fault_time'], 30) <= tmp['dt']).count()
            precision = ntpp / (npp + 1e-20)

            pr_df = fault_tag_df.filter(fault_tag_df['fault_time'] >= test1_begin_date).filter(
                fault_tag_df['fault_time'] < test1_end_date).repartition(1)  # .filter(F.col('model')==1)
            npr = pr_df.count()

            ntpr = pr_df.join(submit, on=['model', 'serial_number'], how='left') \
                .filter(F.col('dt').isNotNull()).filter(F.col('fault_time') >= F.col('dt')).count()
            recall = ntpr / (npr + 1e-20)

            F1_score = 2 * precision * recall / (precision + recall + 1e-10)
            return ({
                'npp': npp,
                'ntpp': ntpp,
                'npr': npr,
                'ntpr': ntpr,
                'precision': precision,
                'recall': recall,
                'F1': F1_score
            })

        def testF1(thresh_spe_lgb):

            submit_spe_lgb = preds_lgb_cat_spark.filter(((F.col('prediction_spe_lgb')) > thresh_spe_lgb)).select(
                ['model', 'serial_number', 'dt']) \
                .repartition(1)

            submit_spe_lgb.registerTempTable('submit_spe_lgb')
            submit_spe_lgb=spark.sql("select model,serial_number,min(dt) as dt from submit_spe_lgb group by model,serial_number")


            fault_tag_df = spark.read.csv('../data/disk_sample_fault_tag.csv', header=True)
            fault_tag_df = fault_tag_df.withColumn('fault_time',
                                                   fault_tag_df['fault_time'].cast(DateType()))
            fault_tag_df.registerTempTable('fault_tag')

            fault_tag_df = spark.sql("select model,serial_number,min(fault_time) as fault_time "
                                     "from fault_tag "
                                     "group by model,serial_number")
            print('spe_lgb:',cal_F1(submit_spe_lgb,fault_tag_df))


        for thresh_spe_lgb in np.linspace(best_thresh_spe_lgb-0.01, best_thresh_spe_lgb + 0.03, 9):
            print(testF1(thresh_spe_lgb))





