"""
Date column inference: used by analytics (trends) and ML (predictions).
The system searches the dataset for a date column in this priority order and uses it
for trends over time and next-year/month predictions. No hardcoded column name.
"""
DATE_COLUMN_PRIORITY = [
    "order_date",
    "order date",
    "transaction_date",
    "date",
    "created_at",
    "timestamp",
]
