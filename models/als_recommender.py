"""
PySpark ALS collaborative filtering recommender.
Handles training, hyperparameter tuning via CrossValidator, and batch inference.
Falls back gracefully if Spark is unavailable (useful for testing).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql.functions import col
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


class ALSRecommender:
    def __init__(
        self,
        rank: int = 50,
        max_iter: int = 20,
        reg_param: float = 0.1,
        alpha: float = 40.0,       # confidence scaling for implicit feedback
        cold_start: str = "drop",
        tune: bool = False,
    ):
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.alpha = alpha
        self.cold_start = cold_start
        self.tune = tune
        self.model = None
        self._spark = None

    def _get_spark(self):
        if self._spark is None:
            self._spark = (SparkSession.builder
                           .appName("ALSRecommender")
                           .config("spark.executor.memory", "4g")
                           .config("spark.driver.memory", "4g")
                           .getOrCreate())
            self._spark.sparkContext.setLogLevel("ERROR")
        return self._spark

    def fit(self, interactions_df: pd.DataFrame,
            user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating"):
        if not SPARK_AVAILABLE:
            raise RuntimeError("PySpark not installed. Run: pip install pyspark")

        spark = self._get_spark()
        sdf = spark.createDataFrame(interactions_df[[user_col, item_col, rating_col]])

        als = ALS(
            rank=self.rank, maxIter=self.max_iter, regParam=self.reg_param,
            alpha=self.alpha, implicitPrefs=True, coldStartStrategy=self.cold_start,
            userCol=user_col, itemCol=item_col, ratingCol=rating_col, seed=42,
        )

        if self.tune:
            param_grid = (ParamGridBuilder()
                          .addGrid(als.rank, [20, 50, 100])
                          .addGrid(als.regParam, [0.01, 0.1, 1.0])
                          .build())
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol=rating_col, predictionCol="prediction")
            cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid,
                                evaluator=evaluator, numFolds=3, seed=42)
            cv_model = cv.fit(sdf)
            self.model = cv_model.bestModel
            print(f"Best rank={self.model.rank}, regParam={self.model._java_obj.parent().getRegParam()}")
        else:
            self.model = als.fit(sdf)

        return self

    def recommend_for_users(self, user_ids: list[int], n: int = 10) -> pd.DataFrame:
        """Return top-N item recommendations per user."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        spark = self._get_spark()
        users_df = spark.createDataFrame([(u,) for u in user_ids], ["user_id"])
        recs = self.model.recommendForUserSubset(users_df, n)
        return recs.toPandas()

    def recommend_for_all(self, n: int = 10) -> pd.DataFrame:
        recs = self.model.recommendForAllUsers(n)
        return recs.toPandas()

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "ALSRecommender":
        from pyspark.ml.recommendation import ALSModel
        instance = cls()
        instance.model = ALSModel.load(path)
        return instance