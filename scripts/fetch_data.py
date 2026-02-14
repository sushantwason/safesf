"""
Fetch 311 data from SF Open Data API and save to local/GCS
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sodapy import Socrata
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SF311DataFetcher:
    """Fetch and process SF 311 service request data"""

    def __init__(self):
        self.domain = os.getenv('SF_OPEN_DATA_DOMAIN', 'data.sfgov.org')
        self.dataset_id = os.getenv('SF_OPEN_DATA_DATASET_ID', 'vw6y-z8j6')
        self.app_token = os.getenv('SF_OPEN_DATA_APP_TOKEN')

        # Initialize Socrata client (token is optional but recommended)
        if self.app_token and self.app_token != 'YOUR_TOKEN_HERE':
            logger.info("Using authenticated API access")
            self.client = Socrata(
                self.domain,
                self.app_token,
                timeout=60
            )
        else:
            logger.warning("No API token found - using unauthenticated access (rate limited)")
            self.client = Socrata(
                self.domain,
                app_token=None,
                timeout=60
            )

    def fetch_data(self, start_date: str, end_date: str, limit: int = 1000000) -> pd.DataFrame:
        """
        Fetch 311 data for date range

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum records to fetch

        Returns:
            DataFrame with 311 cases
        """
        logger.info(f"Fetching 311 data from {start_date} to {end_date}")

        # Build SoQL query (use requested_datetime, not opened)
        where_clause = f"requested_datetime >= '{start_date}T00:00:00' AND requested_datetime <= '{end_date}T23:59:59'"

        try:
            results = self.client.get(
                self.dataset_id,
                where=where_clause,
                limit=limit,
                order="requested_datetime DESC"
            )

            df = pd.DataFrame.from_records(results)
            logger.info(f"Fetched {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess 311 data"""
        logger.info("Cleaning data...")

        # Parse dates (use requested_datetime instead of opened)
        if 'requested_datetime' in df.columns:
            df['opened'] = pd.to_datetime(df['requested_datetime'])
        if 'closed_date' in df.columns:
            df['closed'] = pd.to_datetime(df['closed_date'], errors='coerce')

        # Extract temporal features
        df['year'] = df['opened'].dt.year
        df['month'] = df['opened'].dt.month
        df['day'] = df['opened'].dt.day
        df['hour'] = df['opened'].dt.hour
        df['day_of_week'] = df['opened'].dt.dayofweek
        df['week_of_year'] = df['opened'].dt.isocalendar().week

        # Clean categories (use service_name as category)
        if 'service_name' in df.columns:
            df['category'] = df['service_name'].fillna('Unknown')
        elif 'category' in df.columns:
            df['category'] = df['category'].fillna('Unknown')
        else:
            df['category'] = 'Unknown'

        # Handle coordinates
        if 'point' in df.columns:
            df['latitude'] = df['point'].apply(
                lambda x: float(x['coordinates'][1]) if pd.notna(x) and 'coordinates' in x else None
            )
            df['longitude'] = df['point'].apply(
                lambda x: float(x['coordinates'][0]) if pd.notna(x) and 'coordinates' in x else None
            )

        # Handle supervisor districts
        if 'supervisor_district' in df.columns:
            df['supervisor_district'] = pd.to_numeric(
                df['supervisor_district'], errors='coerce'
            )

        # Calculate resolution time
        if 'closed' in df.columns:
            df['resolution_hours'] = (
                (df['closed'] - df['opened']).dt.total_seconds() / 3600
            )

        logger.info(f"Cleaned data shape: {df.shape}")
        return df

    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save data to parquet file"""
        logger.info(f"Saving data to {output_path}")

        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet for efficiency
        df.to_parquet(output_path, compression='gzip', index=False)

        logger.info(f"Saved {len(df)} records to {output_path}")

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['opened'].min().isoformat(),
                'end': df['opened'].max().isoformat()
            },
            'categories': df['category'].value_counts().head(10).to_dict(),
            'neighborhoods': df.get('neighborhood', pd.Series()).value_counts().head(10).to_dict(),
            'avg_resolution_hours': df.get('resolution_hours', pd.Series()).mean(),
            'missing_location': df[['latitude', 'longitude']].isna().sum().sum()
        }
        return summary


def main():
    parser = argparse.ArgumentParser(description='Fetch SF 311 data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/311_raw.parquet', help='Output file path')
    parser.add_argument('--limit', type=int, default=1000000, help='Max records to fetch')

    args = parser.parse_args()

    # Initialize fetcher
    fetcher = SF311DataFetcher()

    # Fetch data
    df = fetcher.fetch_data(args.start_date, args.end_date, args.limit)

    # Clean data
    df = fetcher.clean_data(df)

    # Save data
    fetcher.save_data(df, args.output)

    # Print summary
    summary = fetcher.get_data_summary(df)
    logger.info("Data Summary:")
    logger.info(f"  Total Records: {summary['total_records']}")
    logger.info(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"  Top Categories: {list(summary['categories'].keys())[:5]}")
    logger.info(f"  Avg Resolution Time: {summary['avg_resolution_hours']:.2f} hours")

    logger.info("✅ Data fetch complete!")


if __name__ == '__main__':
    main()
