import pyspark.sql.functions as F
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)
import os
import json
from datetime import datetime
import time
today = datetime.today()

driver = os.environ.get('driver', 'oracle.jdbc.OracleDriver')
url = os.environ.get('url', '')

host = os.environ.get('host', '')
port = os.environ.get('port', '')
sid = os.environ.get('sid', '')
user = os.environ.get('user', '')
password = os.environ.get('password', '')
table_name = os.environ.get('table_name', '')

from me_recomendation.config.configuration import getConfig, setConfig
    
def getLastFile(fdir, sc):
    hadoop = sc._jvm.org.apache.hadoop
    
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    path = hadoop.fs.Path(fdir)
    files = [str(f.getPath()).split('/')[-1] for f in fs.get(conf).listStatus(path)]
    files = list(set(files))
    files.sort()
    return files[-1]

def getConfig(str1, str2, sc=None):
    from me_recomendation.config.configuration import getConfig as _cfg_get
    try:
        val = _cfg_get(str1, str2)
    except KeyError:
        val = "None"
        
    if val == "None":
        if str2 == "maximum_date":
            return today.strftime("%Y-%m-%d")
        elif str2 == "fe_fpath":
            fdir = "./recommendation_system/pipeline/product_recommendation/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "model_fpath":
            fdir = "./recommendation_system/model/product_recommendation/"
            lastFile = getLastFile(fdir, sc) if sc else "latest"
            return f"{fdir}{lastFile}/best_model_{lastFile}"
        elif str2 == "recency_bucketizer_fpath":
            fdir = "./recommendation_system/pipeline/rfm/recency/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "frequency_bucketizer_fpath":
            fdir = "./recommendation_system/pipeline/rfm/frequency/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "monetary_bucketizer_fpath":
            fdir = "./recommendation_system/pipeline/rfm/monetary/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "age_bucketizer_fpath":
            fdir = "./recommendation_system/pipeline/bucketizer/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "customer_preprocessing_fpath":
            fdir = "./recommendation_system/pipeline/preprocessing_customer/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "clustering_model_fpath":
            fdir = "./recommendation_system/model/clustering/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "simobi_preprocessing_fpath":
            fdir = "./recommendation_system/pipeline/simobi/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "simobi_model_fpath":
            fdir = "./recommendation_system/model/simobi/"
            return fdir + getLastFile(fdir, sc) if sc else fdir
        elif str2 == "assembler_fpath":
            fdir = "./recommendation_system/pipeline/assembler/"
            lastFile = getLastFile(fdir, sc) if sc else "latest"
            return f"{fdir}{lastFile}/assembler_{lastFile}"
        elif str2 == "binarizer_fpath":
            fdir = "./recommendation_system/pipeline/binarizer/"
            lastFile = getLastFile(fdir, sc) if sc else "latest"
            return fdir + lastFile
    return val

def writeData(df, spark, db, table, partition=None):
    if partition:
        try:
            firstPartition = get_partition(spark, table, db, 0)
            firstPartition = datetime.strptime(firstPartition, '%Y-%m-%d')
            lastPartition = get_partition(spark, table, db)
            lastPartition = datetime.strptime(lastPartition, '%Y-%m-%d')
            
            #check data retention
            dataRetention = getConfig('core', 'data_retention')
            dataRange = (lastPartition-firstPartition).total_seconds()
            dataRange = int(divmod(dataRange, 31536000)[0])
            
            #get write mode
            if dataRange>=dataRetention:
                writeMode = 'overwrite'
            else:
                writeMode = 'append'
        except Exception:
            writeMode = 'append'
            
        df.write.format('delta').partitionBy(partition).mode(writeMode).saveAsTable('{}.{}'.format(db,table))
    else:
        df.write.format('delta').mode('overwrite').saveAsTable('{}.{}'.format(db,table))
        
def get_partition(spark, table, db='dm_analytics', position=-1):
    try:
        partition = spark.sql("SHOW PARTITIONS {}.{}".format(db, table))
        p = [p.partition.split('=')[-1] for p in partition.collect()]
        return p[position]
    except Exception:
        return today.strftime('%Y-%m-%d')

def checkAllLastPartition(spark, dt_pr, waitingTime=18000, sleepTime=600):
    # sleepTime = 600 Check every 10 Minutes
    # waitingTime = 18000 wait for 5 hours
    
    allTableReady = False
    waitTime = 0
    listTable = getConfig('core', 'list_table')
    while (allTableReady==False) and (waitTime<waitingTime):
        notUpdatedTable = []
        df = (spark.read.format("jdbc")
              .option("url", url)
              .option("dbtable", table_name)
              .option("user", user)
              .option("password", password)
              .option("driver", driver)
              .load())
        df = df.filter(F.col('XML_SOURCE').contains('SQOOP'))
        df = df.filter(F.col('TRG_TBL_NAME').contains('DM_ANALYTICS'))
        df = df.select("RELATIONAL_TARGET", "START_DATE", "END_DATE", "DT_PR", "PRC_DT")
        df = df.select("TRG_TBL_NAME", "START_DATE", "FINISHED_DATE", "DT_PR", "PRC_DT")
        splitcol = F.split(F.col('TRG_TBL_NAME'), r'\.')
        df = df.withColumn("SCHEMA", splitcol.getItem(0))
        df = df.withColumn("TABLE", splitcol.getItem(1))
        df = df.drop("TRG_TBL_NAME")
        
        dfpd = df.toPandas()
        dfpd.head()
        dfpd['DT_PR'] = pd.to_datetime(dfpd['DT_PR'], format='%Y%m%d') # change prc_dt to dt_pr
        dfpd = dfpd.loc[dfpd.reset_index().groupby('TABLE')['DT_PR'].idxmax()] # change variable check from prc_dt to dt_pr edited by rendi
        dfpd.reset_index(inplace=True, drop=True)
        
        list_table = [x.upper() for x in getConfig('core', 'list_table_check')]
        dfpd = dfpd.loc[dfpd['TABLE'].isin(list_table)].reset_index(drop=True)
        
        toString = lambda x: x.strftime('%Y%m%d')
        dfpd['DT_PR'] = dfpd['DT_PR'].apply(toString) # chnage prc_dt to dt_pr
        
        dfsub = dfpd.loc[dfpd['DT_PR'] < dt_pr, ["TABLE", "DT_PR"]]
        notUpdatedTable = [[x[0], x[1]] for x in dfsub.values]
        
        if len(notUpdatedTable) == 0:
            allTableReady = True
        else:
            waitTime+=sleepTime
            time.sleep(sleepTime)
    df = spark.createDataFrame(dfpd)
    df.write.mode('overwrite').saveAsTable("mldb.mii_rs_table_summary")
    return notUpdatedTable, allTableReady

def model_comparison(spark):
    model_summary = spark.sql('SELECT * FROM mldb.mii_rs_best_model_summary').toPandas()
    max_date = model_summary['training_date'].max()
    model_summary1 = model_summary[model_summary['training_date'] == max_date]
    score1 = model_summary1['test_f1'].mean()
    
    #get previous model
    training_date = getConfig('prediction', 'training_date')
    model_summary = model_summary[model_summary['training_date'] != max_date]
    # max_date = model_summary['training_date'].max()
    model_summary0 = model_summary[model_summary['training_date'] == training_date]
    score0 = model_summary0['test_f1'].mean()
    
    if score1<score0:
        training_date = training_date.replace('-', '')
        value = './recommendation_system/model/product_recommendation/{}/best_model_{}'.format(training_date, training_date)
        setConfig('prediction', 'model_fpath', value, True)
        
        value = './recommendation_system/pipeline/product_recommendation/feature_engineering_{}'.format(training_date)
        setConfig('prediction', 'fe_fpath', value)
        
        value = './recommendation_system/pipeline/rfm/recency/recency_bucketizer_{}'.format(training_date)
        setConfig('prediction', 'recency_bucketizer_fpath', value)
        
        value = './recommendation_system/pipeline/rfm/frequency/frequency_bucketizer_{}'.format(training_date)
        setConfig('prediction', 'frequency_bucketizer_fpath', value)
        
        value = './recommendation_system/pipeline/rfm/monetary/monetary_bucketizer_{}'.format(training_date)
        setConfig('prediction', 'monetary_bucketizer_fpath', value)
        
        value = './recommendation_system/pipeline/bucketizer/age_bucketizer_{}'.format(training_date)
        setConfig('prediction', 'age_bucketizer_fpath', value)
        
        value = './recommendation_system/pipeline/preprocessing_customer/preprocessing_customer_{}'.format(training_date)
        setConfig('prediction', 'customer_preprocessing_fpath', value)
        
        value = './recommendation_system/model/clustering/kmeans_{}'.format(training_date)
        setConfig('prediction', 'clustering_model_fpath', value)
        
        value = './recommendation_system/pipeline/simobi/preprocessing_{}'.format(training_date)
        setConfig('prediction', 'simobi_preprocessing_fpath', value)
        
        value = './recommendation_system/model/simobi/model_{}'.format(training_date)
        setConfig('prediction', 'simobi_model_fpath', value)
        
        value = './recommendation_system/pipeline/binarizer/binarizer_{}'.format(training_date)
        setConfig('prediction', 'binarizer_fpath', value)
    else:
        setConfig('prediction', 'training_date', max_date)
        setConfig('prediction', 'fe_fpath')
        setConfig('prediction', 'model_fpath')
        setConfig('prediction', 'recency_bucketizer_fpath')
        setConfig('prediction', 'frequency_bucketizer_fpath')
        setConfig('prediction', 'monetary_bucketizer_fpath')
        setConfig('prediction', 'age_bucketizer_fpath')
        setConfig('prediction', 'customer_preprocessing_fpath')
        setConfig('prediction', 'clustering_model_fpath')
        setConfig('prediction', 'simobi_preprocessing_fpath')
        setConfig('prediction', 'simobi_model_fpath')
        setConfig('prediction', 'binarizer_fpath')

def validate_csv_file(df, category):
    if (category != 'product_exclution') and (len(df.columns) == 1):
        raise AttributeError("Please use ',' delimiter!")
        
    if (category == "term_and_condition") and any(c not in ['product', 'field', 'value', 'filter_type'] for c in df.columns):
        raise AttributeError("Check the column name!, please use product,field,value,filter_type as the column names")
        
    # Note: in the original source screenshot, this check was written as:
    # if (category == "product_exclution") and (c != 'product'):
    # which raised NameError because `c` was unbound. Written to check columns accurately:
    if (category == "product_exclution") and any(c != 'product' for c in df.columns):
        raise AttributeError("Check the column name!, please use product as the column name!")
        
    if (category == "occupation") and any(c not in ['job_desc', 'occupation'] for c in df.columns):
        raise AttributeError("Check the column name!, please use job_desc and occupation as the column names")
        
def head(df, n=5):
    display(df.limit(n).toPandas())
    
def schema(df):
    df.printSchema()
    
def distinct(df, col):
    display(df.select(col).distinct().toPandas())
    
def clean_text(df, coln):
    df = df.withColumn(coln, F.lower(F.col(coln)))
    df = df.withColumn(coln, F.trim(F.col(coln)))
    df = df.withColumn(coln, F.regexp_replace(F.col(coln), "[^a-z0-9] ", ""))
    df = df.withColumn(coln, F.regexp_replace(F.col(coln), r"\s+", " "))
    return df

def get_summary(df, colname):
    df_summary = df.groupBy(colname).count().toPandas()
    df_summary.sort_values(by='count', inplace=True, ascending=False)
    df_summary['percentage'] = [round(x/df_summary['count'].sum(), 4)*100 for x in df_summary['count'].values]
    df_summary.reset_index(inplace=True, drop=True)
    return df_summary

def data_summary(df, colname = 'gender', var_type='numeric'):
    if var_type=='numeric':
        display(df.describe([colname]).show())
    elif var_type=='category':
        n_unique = df.select(colname).distinct().count()
        print(colname+' have', n_unique, 'unique values')
        df_summary = get_summary(df, colname)
        
        if n_unique<=20:
            display(df_summary)
        else:
            display(df_summary.head(20))
            
def to_none(df):
    data_type = dict(df.dtypes)
    for coln in df.columns:
        if data_type[coln] != 'timestamp':
            df = df.withColumn(coln,
                               F.when((df[coln].isNull()) | (F.isnan(df[coln])) | (df[coln]=='null') | (df[coln]=='') | (df[coln]=='none') | (df[coln]=='nan') | (df[coln]=='NaN') | (df[coln]=='None') | (df[coln]=='Null') | (df[coln]=='NULL'),
                                      F.lit(None)).otherwise(df[coln]))
    return df

def check_missing_values(df):
    n_rows = df.count()
    n_miss_list = []
    perc_miss_list = []
    dimensions = []
    for coln in df.columns:
        n_miss = df.filter(df[coln].isNull()).count()
        n_miss_list.append(n_miss)
        
        perc_miss = round(n_miss/n_rows, 4)*100
        perc_miss_list.append(perc_miss)
        
        n_unique = df.select(coln).distinct().count()
        dimensions.append(n_unique)
    miss_val = pd.DataFrame()
    miss_val['Features'] = df.columns
    miss_val['Number of Missing'] = n_miss_list
    miss_val['Percent Missing'] = perc_miss_list
    miss_val['Dimensions'] = dimensions
    miss_val.sort_values(by='Number of Missing', inplace=True, ascending=False)
    miss_val.reset_index(inplace=True, drop=True)
    display(miss_val)
    
def fill_with_zero(df, col_list):
    #fill missing values
    for coln in col_list:
        df = df.withColumn(coln, F.when((df[coln].isNull()), F.lit(0)).otherwise(df[coln]))
        
    return df

def mode_of_pyspark_columns(df, cat_col_list):
    col_with_mode=[]
    for coln in cat_col_list:
        #filter null
        df = df.filter(df[coln].isNull()==False)
        df_summary = df.groupBy(coln).count().toPandas()
        
        #sort unique values w.r.t their count values
        df_summary.sort_values(by='count', inplace=True, ascending=False)
        
        col_with_mode.append([coln, df_summary[coln].values[0]])
    col_with_mode = pd.DataFrame(col_with_mode, columns = ['column', 'mode'])
    return col_with_mode

def fill_missing_with_mode(df, cat_col_list, train=True):
    if train:
        col_with_mode = mode_of_pyspark_columns(df, cat_col_list)
        try:
            col_with_mode.to_csv('./recommendation_system/dictionary/column_mode.csv', index=False)
        except:
            try:
                col_with_mode.to_csv('../dictionary/column_mode.csv', index=False)
            except:
                col_with_mode.to_csv('../../dictionary/column_mode.csv', index=False)
    else:
        col_mode_fpath = './recommendation_system/dictionary/column_mode.csv'
        col_with_mode = pd.read_csv(col_mode_fpath)
        
    for coln, mode in col_with_mode.values.tolist():
        df = df.fillna(mode, subset=[coln])
    return df

class Performance:
    def __init__(self, labelCol='label', predictCol='prediction', probabilityCol='probability'):
        self.labelCol = labelCol
        self.predictCol = predictCol
        self.probabilityCol = probabilityCol
        
    def getMetrics(self, result):
        cols = [self.predictCol, self.labelCol]
        predictionAndLabels = result.select(*(F.col(c).cast('double').alias(c) for c in cols))
        predictionAndLabels = predictionAndLabels.withColumnRenamed(self.labelCol, 'target_index')
        predictionAndLabels = predictionAndLabels.rdd
        
        scoreM = MulticlassMetrics(predictionAndLabels)
        self.scoreM = scoreM
        
        scoreB = BinaryClassificationMetrics(predictionAndLabels)
        self.scoreB = scoreB
        
    def getAccuracy(self):
        return round(self.scoreM.accuracy, 4)
        
    def getRecall(self):
        return round(self.scoreM.recall(), 4)
        
    def getPrecision(self):
        return round(self.scoreM.precision(), 4)
        
    def getAvgLogloss(self, result):
        result = result.withColumn('logloss',
                                   F.when(F.col(self.labelCol)==1,
                                          F.lit(-F.log(F.col(self.probabilityCol)))).otherwise(
                                          F.lit(-F.log(1-F.col(self.probabilityCol)))))
        avgLogloss = result.select(F.mean('logloss').alias('avglogloss')).collect()[0].avglogloss
        return round(avgLogloss, 4)
        
    def getF1score(self, result):
        evaluator = MulticlassClassificationEvaluator(predictionCol=self.predictCol,
                                                      labelCol=self.labelCol)
        return round(evaluator.evaluate(result), 4)
        
    def getAUC(self, result):
        evaluator = BinaryClassificationEvaluator(rawPredictionCol=self.predictCol,
                                                  labelCol=self.labelCol)
        return round(evaluator.evaluate(result), 4)
