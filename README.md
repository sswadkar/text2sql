# Example Orders Database + DAIL-SQL Implementation

This project includes:

- `order_example_db.sql` — a ready-to-import MySQL schema with realistic e-commerce-style data (users, products, orders, reviews, etc.)
- A Python implementation of DAIL-SQL-style prompting that retrieves relevant example prompts using cosine similarity.

## SQL File

You can import `order_example_db.sql` directly into MySQL Workbench or via command line:

```bash
mysql -u your_user -p your_database < order_example_db.sql
```

This schema is designed to support a variety of SQL query types: joins, filtering, grouping, aggregation, subqueries, and more.

## DAIL-SQL-Style Retrieval with TF-IDF

This project includes a script (`sql_intent_retriever.py`) that implements example retrieval similar to [DAIL-SQL](https://arxiv.org/abs/2308.15363). It uses:

* **TF-IDF vectorization** and **cosine similarity** to select the most relevant question-SQL pairs for a given user prompt.
* A simple **hash-based caching system** that stores the vectorizer and embedding matrix to avoid recomputing unless the example set changes.

This allows the system to efficiently select top-k few-shot examples at runtime and construct high-quality prompts for use with models like Claude 3 Sonnet via AWS Bedrock.

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### `requirements.txt` includes:

* `scikit-learn` — TF-IDF vectorizer and cosine similarity
* `joblib` — caching for embeddings
* `boto3` — call Claude via AWS Bedrock
* `python-dotenv` — load `.env` config with inference ARN and region

## `.env`

```
INFERENCE_ARN=your-bedrock-model-arn
```
