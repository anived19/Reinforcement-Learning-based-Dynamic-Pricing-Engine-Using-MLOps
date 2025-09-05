import pandas as pd
from elasticsearch import Elasticsearch, helpers

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Load CSV
df = pd.read_csv("market_data_elk.csv")

# Convert rows into Elasticsearch documents
actions = [
    {
        "_index": "market-data",
        "_source": row.to_dict()
    }
    for _, row in df.iterrows()
]

helpers.bulk(es, actions)
print("✅ Data uploaded to Elasticsearch index: market-data")
