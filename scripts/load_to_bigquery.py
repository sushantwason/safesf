"""
Load 311 data from local parquet files to BigQuery
"""
import os
import pandas as pd
from google.cloud import bigquery
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_to_bigquery(
    data_path: str,
    project_id: str,
    dataset_id: str,
    table_id: str
):
    """Load parquet data to BigQuery"""

    logger.info(f"Loading data from {data_path} to BigQuery...")

    # Initialize client
    client = bigquery.Client(project=project_id)

    # Read parquet file
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")

    # Prepare data for BigQuery
    # Ensure column names match schema
    column_mapping = {
        'opened': 'opened',
        'closed': 'closed',
        'category': 'category',
        'supervisor_district': 'supervisor_district',
        'neighborhood': 'neighborhood',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'resolution_hours': 'resolution_hours',
        'year': 'year',
        'month': 'month',
        'day': 'day',
        'hour': 'hour',
        'day_of_week': 'day_of_week',
        'week_of_year': 'week_of_year'
    }

    # Add case_id if not present
    if 'case_id' not in df.columns and 'caseid' in df.columns:
        df['case_id'] = df['caseid']
    elif 'case_id' not in df.columns:
        df['case_id'] = df.index.astype(str)

    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df_upload = df[available_cols + ['case_id']].copy()

    # Convert timestamps to datetime
    if 'opened' in df_upload.columns:
        df_upload['opened'] = pd.to_datetime(df_upload['opened'])
    if 'closed' in df_upload.columns:
        df_upload['closed'] = pd.to_datetime(df_upload['closed'])

    # Target table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Configure load job
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ],
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="opened"
        )
    )

    # Load data
    logger.info(f"Loading to {table_ref}...")
    job = client.load_table_from_dataframe(
        df_upload,
        table_ref,
        job_config=job_config
    )

    # Wait for completion
    job.result()

    # Verify
    table = client.get_table(table_ref)
    logger.info(f"Loaded {job.output_rows} rows to {table.full_table_id}")
    logger.info(f"Total rows in table: {table.num_rows}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Load 311 data to BigQuery')
    parser.add_argument('--data', type=str, default='data/311_raw.parquet',
                       help='Path to parquet file')
    parser.add_argument('--project', type=str,
                       default=os.getenv('GCP_PROJECT_ID', 'safesf-439219'),
                       help='GCP project ID')
    parser.add_argument('--dataset', type=str,
                       default=os.getenv('GCP_DATASET', 'sf_311_data'),
                       help='BigQuery dataset ID')
    parser.add_argument('--table', type=str, default='requests_raw',
                       help='BigQuery table ID')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Run fetch_data.py first to download 311 data")
        return

    load_to_bigquery(
        args.data,
        args.project,
        args.dataset,
        args.table
    )

    logger.info("✅ Data load complete!")


if __name__ == '__main__':
    main()
