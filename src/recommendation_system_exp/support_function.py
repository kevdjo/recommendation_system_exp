import os
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

today = datetime.today()

from recommendation_system_exp.config.configuration import (
    ARTIFACT_ROOT,
    OUTPUT_DB,
    SOURCE_DB,
    setConfig,
)


def getLastFile(fdir, sc=None):
    # Migration: Serverless (Spark Connect) has no SparkContext/JVM access, so the
    # Hadoop FileSystem listing is replaced by a plain listing of the UC Volume
    # (FUSE-mounted under /Volumes). Same result: lexicographically latest entry.
    files = [f.rstrip("/").split("/")[-1] for f in os.listdir(fdir)]
    files = list(set(files))
    files.sort()
    return files[-1]


def getConfig(str1, str2, sc=None):
    from recommendation_system_exp.config.configuration import getConfig as _cfg_get

    try:
        val = _cfg_get(str1, str2)
    except KeyError:
        val = "None"

    if val == "None":
        if str2 == "fe_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/product_recommendation/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "model_fpath":
            fdir = f"{ARTIFACT_ROOT}/model/product_recommendation/"
            lastFile = getLastFile(fdir, sc)
            return f"{fdir}{lastFile}/best_model_{lastFile}"
        elif str2 == "recency_bucketizer_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/rfm/recency/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "frequency_bucketizer_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/rfm/frequency/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "monetary_bucketizer_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/rfm/monetary/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "age_bucketizer_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/bucketizer/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "customer_preprocessing_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/preprocessing_customer/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "clustering_model_fpath":
            fdir = f"{ARTIFACT_ROOT}/model/clustering/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "mobi_preprocessing_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/mobi/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "mobi_model_fpath":
            fdir = f"{ARTIFACT_ROOT}/model/mobi/"
            return fdir + getLastFile(fdir, sc)
        elif str2 == "assembler_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/assembler/"
            lastFile = getLastFile(fdir, sc)
            return f"{fdir}{lastFile}/assembler_{lastFile}"
        elif str2 == "binarizer_fpath":
            fdir = f"{ARTIFACT_ROOT}/pipeline/binarizer/"
            lastFile = getLastFile(fdir, sc)
            return fdir + lastFile
    return val


def writeData(df, spark, db, table, partition=None):
    if partition:
        try:
            firstPartition = get_partition(spark, table, db, 0)
            firstPartition = datetime.strptime(firstPartition, "%Y-%m-%d")
            lastPartition = get_partition(spark, table, db)
            lastPartition = datetime.strptime(lastPartition, "%Y-%m-%d")

            # check data retention
            dataRetention = getConfig("core", "data_retention")
            dataRange = (lastPartition - firstPartition).total_seconds()
            dataRange = int(divmod(dataRange, 31536000)[0])

            # get write mode
            if dataRange >= dataRetention:
                writeMode = "overwrite"
            else:
                writeMode = "append"
        except Exception:
            writeMode = "append"

        df.write.format("delta").partitionBy(partition).mode(writeMode).saveAsTable(
            f"{db}.{table}"
        )
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(f"{db}.{table}")


def get_partition(spark, table, db=SOURCE_DB, position=-1):
    try:
        partition = spark.sql(f"SHOW PARTITIONS {db}.{table}")
        p = [p.partition.split("=")[-1] for p in partition.collect()]
        return p[position]
    except Exception:
        return today.strftime("%Y-%m-%d")


def model_comparison(spark):
    model_summary = spark.sql(
        f"SELECT * FROM {OUTPUT_DB}.mii_rs_best_model_summary"
    ).toPandas()
    max_date = model_summary["training_date"].max()
    model_summary1 = model_summary[model_summary["training_date"] == max_date]
    score1 = model_summary1["test_f1"].mean()

    # get previous model
    training_date = getConfig("prediction", "training_date")
    model_summary = model_summary[model_summary["training_date"] != max_date]
    # max_date = model_summary['training_date'].max()
    model_summary0 = model_summary[model_summary["training_date"] == training_date]
    score0 = model_summary0["test_f1"].mean()

    if score1 < score0:
        training_date = training_date.replace("-", "")
        value = f"{ARTIFACT_ROOT}/model/product_recommendation/{training_date}/best_model_{training_date}"
        setConfig("prediction", "model_fpath", value, True)

        value = f"{ARTIFACT_ROOT}/pipeline/product_recommendation/feature_engineering_{training_date}"
        setConfig("prediction", "fe_fpath", value)

        value = (
            f"{ARTIFACT_ROOT}/pipeline/rfm/recency/recency_bucketizer_{training_date}"
        )
        setConfig("prediction", "recency_bucketizer_fpath", value)

        value = f"{ARTIFACT_ROOT}/pipeline/rfm/frequency/frequency_bucketizer_{training_date}"
        setConfig("prediction", "frequency_bucketizer_fpath", value)

        value = (
            f"{ARTIFACT_ROOT}/pipeline/rfm/monetary/monetary_bucketizer_{training_date}"
        )
        setConfig("prediction", "monetary_bucketizer_fpath", value)

        value = f"{ARTIFACT_ROOT}/pipeline/bucketizer/age_bucketizer_{training_date}"
        setConfig("prediction", "age_bucketizer_fpath", value)

        value = f"{ARTIFACT_ROOT}/pipeline/preprocessing_customer/preprocessing_customer_{training_date}"
        setConfig("prediction", "customer_preprocessing_fpath", value)

        value = f"{ARTIFACT_ROOT}/model/clustering/kmeans_{training_date}"
        setConfig("prediction", "clustering_model_fpath", value)

        value = f"{ARTIFACT_ROOT}/pipeline/mobi/preprocessing_{training_date}"
        setConfig("prediction", "mobi_preprocessing_fpath", value)

        value = f"{ARTIFACT_ROOT}/model/mobi/model_{training_date}"
        setConfig("prediction", "mobi_model_fpath", value)

        value = f"{ARTIFACT_ROOT}/pipeline/binarizer/binarizer_{training_date}"
        setConfig("prediction", "binarizer_fpath", value)
    else:
        setConfig("prediction", "training_date", max_date)
        setConfig("prediction", "fe_fpath")
        setConfig("prediction", "model_fpath")
        setConfig("prediction", "recency_bucketizer_fpath")
        setConfig("prediction", "frequency_bucketizer_fpath")
        setConfig("prediction", "monetary_bucketizer_fpath")
        setConfig("prediction", "age_bucketizer_fpath")
        setConfig("prediction", "customer_preprocessing_fpath")
        setConfig("prediction", "clustering_model_fpath")
        setConfig("prediction", "mobi_preprocessing_fpath")
        setConfig("prediction", "mobi_model_fpath")
        setConfig("prediction", "binarizer_fpath")


def clean_text(df, coln):
    df = df.withColumn(coln, F.lower(F.col(coln)))
    df = df.withColumn(coln, F.trim(F.col(coln)))
    df = df.withColumn(coln, F.regexp_replace(F.col(coln), "[^a-z0-9] ", ""))
    df = df.withColumn(coln, F.regexp_replace(F.col(coln), r"\s+", " "))
    return df


def get_summary(df, colname):
    df_summary = df.groupBy(colname).count().toPandas()
    df_summary.sort_values(by="count", inplace=True, ascending=False)
    df_summary["percentage"] = [
        round(x / df_summary["count"].sum(), 4) * 100
        for x in df_summary["count"].values
    ]
    df_summary.reset_index(inplace=True, drop=True)
    return df_summary


def to_none(df):
    data_type = dict(df.dtypes)
    for coln in df.columns:
        if data_type[coln] != "timestamp":
            df = df.withColumn(
                coln,
                F.when(
                    (df[coln].isNull())
                    | (F.isnan(df[coln]))
                    | (df[coln] == "null")
                    | (df[coln] == "")
                    | (df[coln] == "none")
                    | (df[coln] == "nan")
                    | (df[coln] == "NaN")
                    | (df[coln] == "None")
                    | (df[coln] == "Null")
                    | (df[coln] == "NULL"),
                    F.lit(None),
                ).otherwise(df[coln]),
            )
    return df


def mode_of_pyspark_columns(df, cat_col_list):
    col_with_mode = []
    for coln in cat_col_list:
        # filter null
        df = df.filter(df[coln].isNull() == False)
        df_summary = df.groupBy(coln).count().toPandas()

        # sort unique values w.r.t their count values
        df_summary.sort_values(by="count", inplace=True, ascending=False)

        col_with_mode.append([coln, df_summary[coln].values[0]])
    col_with_mode = pd.DataFrame(col_with_mode, columns=["column", "mode"])
    return col_with_mode


def fill_missing_with_mode(df, cat_col_list, train=True):
    # Migration: the legacy relative paths do not exist on Serverless; persist
    # the dictionary under the artifact root instead.
    col_mode_fpath = f"{ARTIFACT_ROOT}/dictionary/column_mode.csv"
    if train:
        col_with_mode = mode_of_pyspark_columns(df, cat_col_list)
        os.makedirs(os.path.dirname(col_mode_fpath), exist_ok=True)
        col_with_mode.to_csv(col_mode_fpath, index=False)
    else:
        col_with_mode = pd.read_csv(col_mode_fpath)

    for coln, mode in col_with_mode.values.tolist():
        df = df.fillna(mode, subset=[coln])
    return df


class Performance:
    def __init__(
        self, labelCol="label", predictCol="prediction", probabilityCol="probability"
    ):
        self.labelCol = labelCol
        self.predictCol = predictCol
        self.probabilityCol = probabilityCol

    # Migration: the legacy implementation used pyspark.mllib MulticlassMetrics on
    # an RDD, which Spark Connect (Serverless) does not support. In Spark 2.0 the
    # no-argument MulticlassMetrics.precision()/recall() returned the overall
    # (micro-averaged) value, which equals accuracy, so the same numbers are
    # reproduced with the DataFrame-based evaluator.
    def getMetrics(self, result):
        cols = [self.predictCol, self.labelCol]
        self._scored = result.select(*(F.col(c).cast("double").alias(c) for c in cols))

    def _micro(self):
        evaluator = MulticlassClassificationEvaluator(
            predictionCol=self.predictCol,
            labelCol=self.labelCol,
            metricName="accuracy",
        )
        return evaluator.evaluate(self._scored)

    def getAccuracy(self):
        return round(self._micro(), 4)

    def getRecall(self):
        return round(self._micro(), 4)

    def getPrecision(self):
        return round(self._micro(), 4)

    def getAvgLogloss(self, result):
        result = result.withColumn(
            "logloss",
            F.when(
                F.col(self.labelCol) == 1, F.lit(-F.log(F.col(self.probabilityCol)))
            ).otherwise(F.lit(-F.log(1 - F.col(self.probabilityCol)))),
        )
        avgLogloss = (
            result.select(F.mean("logloss").alias("avglogloss")).collect()[0].avglogloss
        )
        return round(avgLogloss, 4)

    def getF1score(self, result):
        # Spark 4's evaluator tries to read a "probability" column as a vector if one
        # exists; callers overwrite it with a float, so keep only the needed columns.
        result = result.select(self.predictCol, self.labelCol)
        evaluator = MulticlassClassificationEvaluator(
            predictionCol=self.predictCol, labelCol=self.labelCol
        )
        return round(evaluator.evaluate(result), 4)

    def getAUC(self, result):
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol=self.predictCol, labelCol=self.labelCol
        )
        return round(evaluator.evaluate(result), 4)
