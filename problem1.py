from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min, max, count, expr
from pyspark.sql.window import Window

def compute_summary_statistics(spark, input_file):
    # Load data from the input file
    data = spark.read.option("delimiter", "\t").csv(input_file, header=False, inferSchema=True)
    data = data.withColumnRenamed("_c0", "id").withColumnRenamed("_c1", "group").withColumnRenamed("_c2", "value")

    # Compute summary statistics
    summary_stats = data.select(
        mean("value").alias("mean"),
        stddev("value").alias("stddev"),
        min("value").alias("min_value"),
        max("value").alias("max_value"),
        count("value").alias("total_records")
    ).collect()[0]

    # Compute histogram
    num_bins = 10
    bin_width = (summary_stats["max_value"] - summary_stats["min_value"]) / num_bins

    histogram = data.select(
        ((col("value") - summary_stats["min_value"]) / bin_width).cast("int").alias("bin")
    ).groupBy("bin").agg(count("bin").alias("bin_count")).orderBy("bin")

    # Compute median
    window_spec = Window.orderBy("value")
    sorted_data = data.withColumn("row_number", expr("row_number() over (order by value)"))
    total_records = summary_stats["total_records"]
    median_row = total_records // 2

    median_value = sorted_data.filter(col("row_number") == median_row).select("value").collect()[0][0]

    # Print results
    print("Summary Statistics:")
    print(f"Mean: {summary_stats['mean']}")
    print(f"Standard Deviation: {summary_stats['stddev']}")
    print(f"Min: {summary_stats['min_value']}")
    print(f"Max: {summary_stats['max_value']}")
    print(f"Total Records: {summary_stats['total_records']}\n")

    print("Histogram:")
    histogram.show(truncate=False)

    print("\nMedian:")
    print(f"Approximate Median: {median_value}")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    # Specify the input file
    input_file = "data-assignment-8-1M.dat"

    # Specify the number of cores
    num_cores = 2 

    # Create a Spark session with the specified number of cores
    spark = SparkSession.builder.appName("SummaryStatistics").master(f"local[{num_cores}]").getOrCreate()

    # Run the summary statistics and histogram computation
    compute_summary_statistics(spark, input_file)
