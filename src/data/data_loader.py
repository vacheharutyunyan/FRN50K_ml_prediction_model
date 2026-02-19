# src/data/data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
import yaml
from sqlalchemy import create_engine

# Set up logging to help students debug issues
logging.basicConfig(filename="app.log",filemode="a",level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshRetailDataLoader:
    """
    Production-ready data loader for FreshRetailNet-50K dataset.
    
    This class demonstrates best practices for data loading in ML pipelines:
    - Error handling and logging
    - Data validation
    - Caching for efficiency
    - Clear documentation
    """
    
    def __init__(self, config: Optional[dict] = None, config_path: str = "config/config.yaml"):
        """Initialize the data loader with configuration.

        Accepts either a config dict or a path to a yaml file. Initializes
        an SQLite engine (file path can be overridden in config under
        `data.db_path`).
        """
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        self.data_config = self.config.get('data', {})
        self.dataset = None
        self.train_data = None
        self.eval_data = None

        db_path = self.data_config.get('db_path', 'frn50k_data.db')
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        logger.info(f"SQLite engine initialized at: {self.engine.url}")
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the FreshRetailNet-50K dataset from HuggingFace.
        
        Returns:
            Tuple of (train_df, eval_df)
            
        Teaching note: This method demonstrates proper error handling
        and informative logging for production systems.
        """
        try:
            logger.info("Loading FreshRetailNet-50K dataset from files...")

            # Convert to pandas DataFrames for easier manipulation
            train_path = self.data_config.get('train_path')
            eval_path = self.data_config.get('eval_path')
            frac = float(self.data_config.get('fracture', 1.0))

            self.train_data = pd.read_parquet(train_path).sample(frac=frac, random_state=42)
            self.eval_data = pd.read_csv(eval_path).sample(frac=frac, random_state=42)
            
            # Convert datetime column to proper datetime type
            datetime_col = self.data_config.get('datetime_column')
            if datetime_col is None:
                raise KeyError('datetime_column not found in data configuration')
            self.train_data[datetime_col] = pd.to_datetime(self.train_data[datetime_col])
            self.eval_data[datetime_col] = pd.to_datetime(self.eval_data[datetime_col])
            
            logger.info(f"Successfully loaded:")
            logger.info(f"  Training samples: {len(self.train_data):,}")
            logger.info(f"  Evaluation samples: {len(self.eval_data):,}")
            logger.info(f"  Date range: {self.train_data[datetime_col].min()} to {self.train_data[datetime_col].max()}")
            
            # Load data into SQLite database for SQL queries
            logger.info("Loading data into SQLite database...")
            self.train_data.to_sql('train_data', con=self.engine, if_exists='replace', index=False)
            self.eval_data.to_sql('eval_data', con=self.engine, if_exists='replace', index=False)
            logger.info("Data loaded into SQLite. Use SQL queries on: train_data, eval_data tables")
            
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            raise
    
    def get_data_summary(self) -> dict:
        """
        Generate comprehensive data summary for exploratory analysis.
        
        Teaching point: Always start with data understanding before modeling.
        This method provides the foundation for EDA notebooks.
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        datetime_col = self.data_config['datetime_column']
        target_col = self.data_config['target_column']
        
        summary = {
            'dataset_shape': {
                'train': self.train_data.shape,
                'eval': self.eval_data.shape
            },
            'time_span': {
                'start': str(self.train_data[datetime_col].min()),
                'end': str(self.train_data[datetime_col].max()),
                'duration_days': (self.train_data[datetime_col].max() - self.train_data[datetime_col].min()).days
            },
            'business_dimensions': {
                'unique_stores': self.train_data['store_id'].nunique(),
                'unique_products': self.train_data['product_id'].nunique(),
                'unique_cities': self.train_data['city_id'].nunique(),
                'total_store_product_combinations': len(self.train_data.groupby(['store_id', 'product_id']))
            },
            'target_statistics': {
                'mean_sales': self.train_data[target_col].mean(),
                'median_sales': self.train_data[target_col].median(),
                'zero_sales_percentage': (self.train_data[target_col] == 0).mean() * 100,
                'max_sales': self.train_data[target_col].max()
            },
            'stockout_analysis': self._compute_stockout_analysis(),
            'data_quality': {
                'missing_values': self.train_data.isnull().sum().to_dict(),
                'duplicate_rows': self._count_duplicate_rows()
            }
        }

        return summary

    def _compute_stockout_analysis(self) -> dict:
        """
        Compute stockout analysis handling array-like hours_stock_status column.

        The hours_stock_status column may contain arrays/lists of hourly status values.
        """
        if self.train_data is None or self.train_data.empty:
            return {
                'total_observations': 0,
                'stockout_hours': 0,
                'stockout_rate_percent': 0.0,
                'stores_with_stockouts': 0
            }

        total_observations = len(self.train_data)
        stock_status = self.train_data.get('hours_stock_status')
        if stock_status is None or stock_status.empty:
            return {
                'total_observations': total_observations,
                'stockout_hours': 0,
                'stockout_rate_percent': 0.0,
                'stores_with_stockouts': 0
            }

        # Check if the column contains array-like values
        first_value = stock_status.dropna().iloc[0] if not stock_status.dropna().empty else None
        if isinstance(first_value, (list, np.ndarray)):
            # Handle array-like values: count rows where any hour has stockout (0)
            stockout_mask = stock_status.apply(lambda x: (0 in x) if isinstance(x, (list, np.ndarray)) else (x == 0))
            stockout_hours = stock_status.apply(lambda x: sum(1 for v in x if v == 0) if isinstance(x, (list, np.ndarray)) else (1 if x == 0 else 0)).sum()
        else:
            # Handle scalar values
            stockout_mask = stock_status == 0
            stockout_hours = int(stockout_mask.sum())

        return {
            'total_observations': total_observations,
            'stockout_hours': int(stockout_hours),
            'stockout_rate_percent': float(stockout_mask.mean() * 100),
            'stores_with_stockouts': self.train_data[stockout_mask]['store_id'].nunique()
        }

    def _count_duplicate_rows(self) -> int:
        """
        Count duplicate rows, excluding columns with unhashable types (arrays/lists).
        """
        if self.train_data is None or self.train_data.empty:
            return 0

        # Get columns with hashable/simple types only by inspecting first non-null value
        hashable_cols = []
        for col in self.train_data.columns:
            non_null = self.train_data[col].dropna()
            if non_null.empty:
                # treat empty/NaN column as hashable for duplication purposes
                hashable_cols.append(col)
                continue
            first_val = non_null.iloc[0]
            if not isinstance(first_val, (list, np.ndarray)):
                hashable_cols.append(col)

        if hashable_cols:
            return int(self.train_data[hashable_cols].duplicated().sum())
        return 0