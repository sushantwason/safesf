"""
Train ML models for 311 request prediction
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SF311Predictor:
    """ML models for predicting 311 request volumes"""

    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.categories = []

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load preprocessed 311 data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} records")
        return df

    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time series features for prediction"""
        logger.info("Creating time series features...")

        # Group by date and category for volume prediction
        daily_volumes = df.groupby([
            df['opened'].dt.date, 'category'
        ]).size().reset_index()
        daily_volumes.columns = ['date', 'category', 'volume']
        daily_volumes['date'] = pd.to_datetime(daily_volumes['date'])

        # Add temporal features
        daily_volumes['year'] = daily_volumes['date'].dt.year
        daily_volumes['month'] = daily_volumes['date'].dt.month
        daily_volumes['day'] = daily_volumes['date'].dt.day
        daily_volumes['day_of_week'] = daily_volumes['date'].dt.dayofweek
        daily_volumes['week_of_year'] = daily_volumes['date'].dt.isocalendar().week
        daily_volumes['is_weekend'] = daily_volumes['day_of_week'].isin([5, 6]).astype(int)
        daily_volumes['quarter'] = daily_volumes['date'].dt.quarter

        # Lag features (previous days' volumes)
        for category in daily_volumes['category'].unique():
            mask = daily_volumes['category'] == category
            for lag in [1, 7, 14, 30]:
                daily_volumes.loc[mask, f'lag_{lag}d'] = daily_volumes.loc[mask, 'volume'].shift(lag)

        # Rolling averages
        for category in daily_volumes['category'].unique():
            mask = daily_volumes['category'] == category
            for window in [7, 14, 30]:
                daily_volumes.loc[mask, f'rolling_mean_{window}d'] = \
                    daily_volumes.loc[mask, 'volume'].rolling(window=window, min_periods=1).mean()

        # Drop rows with NaN from lag features
        daily_volumes = daily_volumes.dropna()

        logger.info(f"Created features, shape: {daily_volumes.shape}")
        return daily_volumes

    def train_xgboost_model(self, X_train, y_train, X_val, y_val, category: str):
        """Train XGBoost model for a specific category"""
        logger.info(f"Training XGBoost for category: {category}")

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        logger.info(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

    def train_prophet_model(self, df: pd.DataFrame, category: str):
        """Train Prophet model for time series forecasting"""
        logger.info(f"Training Prophet for category: {category}")

        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[df['category'] == category][['date', 'volume']].copy()
        prophet_df.columns = ['ds', 'y']

        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        # Make predictions on validation set (last 30 days)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Calculate metrics on last 30 days
        val_actual = prophet_df.tail(30)['y'].values
        val_pred = forecast.tail(30)['yhat'].values
        mae = mean_absolute_error(val_actual, val_pred)

        logger.info(f"  Prophet MAE: {mae:.2f}")

        return model, {'mae': mae}

    def train_all_models(self, df: pd.DataFrame, output_dir: str = 'models'):
        """Train models for all categories"""
        logger.info("Starting model training...")

        # Create time series features
        ts_df = self.create_time_series_features(df)

        # Get top categories by volume
        top_categories = df['category'].value_counts().head(10).index.tolist()
        self.categories = top_categories

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_metrics = {}

        for category in top_categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for: {category}")
            logger.info(f"{'='*60}")

            # Filter data for this category
            category_df = ts_df[ts_df['category'] == category].copy()

            if len(category_df) < 100:
                logger.warning(f"Insufficient data for {category}, skipping...")
                continue

            # Prepare features
            feature_cols = [col for col in category_df.columns
                          if col not in ['date', 'category', 'volume']]
            self.feature_columns = feature_cols

            X = category_df[feature_cols].values
            y = category_df['volume'].values

            # Time series split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train XGBoost
            xgb_model, xgb_metrics = self.train_xgboost_model(
                X_train, y_train, X_val, y_val, category
            )

            # Train Prophet
            prophet_model, prophet_metrics = self.train_prophet_model(df, category)

            # Save models
            safe_category = category.replace('/', '_').replace(' ', '_')
            joblib.dump(xgb_model, f"{output_dir}/xgb_{safe_category}.pkl")
            joblib.dump(prophet_model, f"{output_dir}/prophet_{safe_category}.pkl")

            all_metrics[category] = {
                'xgboost': xgb_metrics,
                'prophet': prophet_metrics,
                'sample_count': len(category_df)
            }

        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'categories': self.categories,
            'feature_columns': self.feature_columns,
            'metrics': all_metrics
        }

        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("\n✅ Model training complete!")
        logger.info(f"Models saved to: {output_dir}")

        return metadata


def main():
    parser = argparse.ArgumentParser(description='Train 311 prediction models')
    parser.add_argument('--data', type=str, default='data/311_raw.parquet',
                       help='Input data file')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')

    args = parser.parse_args()

    # Initialize predictor
    predictor = SF311Predictor()

    # Load data
    df = predictor.load_data(args.data)

    # Train models
    metadata = predictor.train_all_models(df, args.output)

    # Print summary
    logger.info("\n📊 Training Summary:")
    logger.info(f"Categories trained: {len(metadata['categories'])}")
    logger.info(f"Features used: {len(metadata['feature_columns'])}")

    for category, metrics in metadata['metrics'].items():
        logger.info(f"\n{category}:")
        logger.info(f"  XGBoost MAE: {metrics['xgboost']['mae']:.2f}")
        logger.info(f"  Prophet MAE: {metrics['prophet']['mae']:.2f}")
        logger.info(f"  Sample count: {metrics['sample_count']}")


if __name__ == '__main__':
    main()
