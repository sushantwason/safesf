"""
Set up BigQuery tables for SF 311 data
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset(client: bigquery.Client, dataset_id: str, location: str = "us-west1"):
    """Create BigQuery dataset if it doesn't exist"""
    dataset_ref = f"{client.project}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
        return client.get_dataset(dataset_ref)
    except NotFound:
        logger.info(f"Creating dataset {dataset_id}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset.description = "SF 311 service request data and predictions"
        dataset = client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {dataset.dataset_id}")
        return dataset


def create_311_raw_table(client: bigquery.Client, dataset_id: str):
    """Create table for raw 311 data"""
    table_id = f"{client.project}.{dataset_id}.requests_raw"

    schema = [
        bigquery.SchemaField("case_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("opened", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("closed", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("subcategory", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("neighborhood", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("supervisor_district", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("address", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("resolution_hours", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("year", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("month", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("day", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("hour", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("day_of_week", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("week_of_year", "INTEGER", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.description = "Raw SF 311 service request data"
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="opened"
    )
    table.clustering_fields = ["category", "neighborhood"]

    try:
        client.get_table(table_id)
        logger.info(f"Table {table_id} already exists")
    except NotFound:
        table = client.create_table(table)
        logger.info(f"Created table {table.table_id}")


def create_predictions_table(client: bigquery.Client, dataset_id: str):
    """Create table for predictions"""
    table_id = f"{client.project}.{dataset_id}.predictions"

    schema = [
        bigquery.SchemaField("prediction_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("prediction_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("predicted_volume", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("lower_bound", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("upper_bound", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("model_used", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.description = "311 request volume predictions"
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="prediction_date"
    )
    table.clustering_fields = ["category", "model_used"]

    try:
        client.get_table(table_id)
        logger.info(f"Table {table_id} already exists")
    except NotFound:
        table = client.create_table(table)
        logger.info(f"Created table {table.table_id}")


def create_aggregated_table(client: bigquery.Client, dataset_id: str):
    """Create table for aggregated daily stats"""
    table_id = f"{client.project}.{dataset_id}.daily_aggregates"

    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("total_requests", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("avg_resolution_hours", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("neighborhood_count", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("weekend", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.description = "Daily aggregated 311 statistics"
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="date"
    )
    table.clustering_fields = ["category"]

    try:
        client.get_table(table_id)
        logger.info(f"Table {table_id} already exists")
    except NotFound:
        table = client.create_table(table)
        logger.info(f"Created table {table.table_id}")


def create_views(client: bigquery.Client, dataset_id: str):
    """Create useful views"""

    # View: Recent requests by category
    view_id = f"{client.project}.{dataset_id}.recent_requests_by_category"
    view_query = f"""
    SELECT
        category,
        COUNT(*) as request_count,
        AVG(resolution_hours) as avg_resolution_hours,
        DATE(opened) as date
    FROM `{client.project}.{dataset_id}.requests_raw`
    WHERE opened >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    GROUP BY category, DATE(opened)
    ORDER BY date DESC, request_count DESC
    """

    view = bigquery.Table(view_id)
    view.view_query = view_query

    try:
        client.get_table(view_id)
        logger.info(f"View {view_id} already exists")
    except NotFound:
        view = client.create_table(view)
        logger.info(f"Created view {view.table_id}")


def main():
    """Set up all BigQuery resources"""
    project_id = os.getenv("GCP_PROJECT_ID", "safesf-439219")
    dataset_id = os.getenv("GCP_DATASET", "sf_311_data")
    location = os.getenv("GCP_REGION", "us-west1")

    logger.info(f"Setting up BigQuery for project: {project_id}")

    # Initialize client
    client = bigquery.Client(project=project_id)

    # Create dataset
    create_dataset(client, dataset_id, location)

    # Create tables
    logger.info("Creating tables...")
    create_311_raw_table(client, dataset_id)
    create_predictions_table(client, dataset_id)
    create_aggregated_table(client, dataset_id)

    # Create views
    logger.info("Creating views...")
    create_views(client, dataset_id)

    logger.info("\n✅ BigQuery setup complete!")
    logger.info(f"Dataset: {project_id}.{dataset_id}")
    logger.info("Tables created:")
    logger.info(f"  - {dataset_id}.requests_raw")
    logger.info(f"  - {dataset_id}.predictions")
    logger.info(f"  - {dataset_id}.daily_aggregates")
    logger.info("Views created:")
    logger.info(f"  - {dataset_id}.recent_requests_by_category")


if __name__ == "__main__":
    main()
