"""PySpark batch ingestion: raw CSV -> bronze -> silver, with quarantine for bad records."""
import argparse
import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingestion")

RAW_SCHEMA = StructType([
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("product_id", StringType()),
    StructField("quantity", StringType()),
    StructField("unit_price", StringType()),
    StructField("order_ts", StringType()),
    StructField("status", StringType()),
    StructField("country", StringType()),
    StructField("_corrupt_record", StringType()),
])


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("orders-ingestion")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


def read_raw(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(RAW_SCHEMA)
        .load(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
        .cache()
    )


def clean_and_type(df: DataFrame) -> DataFrame:
    return df.select(
        F.trim("order_id").alias("order_id"),
        F.trim("customer_id").alias("customer_id"),
        F.upper(F.trim("product_id")).alias("product_id"),
        F.col("quantity").cast(IntegerType()).alias("quantity"),
        F.col("unit_price").cast(DoubleType()).alias("unit_price"),
        F.to_timestamp("order_ts").alias("order_ts"),
        F.lower(F.trim("status")).alias("status"),
        F.upper(F.trim("country")).alias("country"),
        "_source_file",
        "_ingested_at",
    )


def apply_quality_rules(df: DataFrame):
    rules = {
        "missing_order_id": F.col("order_id").isNull() | (F.col("order_id") == ""),
        "invalid_quantity": F.col("quantity").isNull() | (F.col("quantity") <= 0),
        "invalid_price": F.col("unit_price").isNull() | (F.col("unit_price") < 0),
        "invalid_timestamp": F.col("order_ts").isNull(),
    }
    failures = F.array_remove(
        F.array(*[F.when(cond, F.lit(name)) for name, cond in rules.items()]), None
    )
    flagged = df.withColumn("_dq_failures", failures)
    valid = flagged.filter(F.size("_dq_failures") == 0).drop("_dq_failures")
    rejected = flagged.filter(F.size("_dq_failures") > 0)
    return valid, rejected


def deduplicate(df: DataFrame, key: str = "order_id") -> DataFrame:
    w = Window.partitionBy(key).orderBy(F.col("order_ts").desc(), F.col("_ingested_at").desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def enrich(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("total_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("order_date", F.to_date("order_ts"))
    )


def write_parquet(df: DataFrame, path: str, partition_col: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).format("parquet").partitionBy(partition_col).save(path)


def run_batch(spark: SparkSession, base_path: str) -> None:
    raw = read_raw(spark, f"{base_path}/raw/orders")
    total = raw.count()

    malformed = raw.filter(F.col("_corrupt_record").isNotNull())
    parsed = raw.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

    bronze = parsed.withColumn("_ingest_date", F.to_date("_ingested_at"))
    write_parquet(bronze, f"{base_path}/bronze/orders", "_ingest_date")

    valid, rejected = apply_quality_rules(clean_and_type(parsed))
    silver = enrich(deduplicate(valid))
    write_parquet(silver, f"{base_path}/silver/orders", "order_date")

    quarantine = malformed.select(
        F.col("_corrupt_record").alias("payload"),
        F.array(F.lit("malformed_record")).alias("_dq_failures"),
        "_source_file", "_ingested_at",
    ).unionByName(
        rejected.select(
            F.to_json(F.struct("order_id", "customer_id", "product_id", "quantity",
                               "unit_price", "order_ts", "status", "country")).alias("payload"),
            "_dq_failures", "_source_file", "_ingested_at",
        )
    )
    quarantine = quarantine.withColumn("_ingest_date", F.to_date("_ingested_at"))
    write_parquet(quarantine, f"{base_path}/quarantine/orders", "_ingest_date", mode="append")

    n_silver, n_quarantine = silver.count(), quarantine.count()
    log.info("read=%d silver=%d quarantined=%d", total, n_silver, n_quarantine)
    if total and n_quarantine / total > 0.20:
        raise RuntimeError(f"Quarantine rate {n_quarantine / total:.0%} exceeds 20% threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="./data")
    args = parser.parse_args()

    spark = get_spark()
    try:
        run_batch(spark, args.base_path)
    finally:
        spark.stop()
