import hashlib
import json
import boto3
import os
import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CACHE_DIR = ".cache"
HASH_FILE = os.path.join(CACHE_DIR, "example_hash.txt")
VEC_FILE = os.path.join(CACHE_DIR, "vectorizer.pkl")
MAT_FILE = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")

INF_ARN = os.getenv("INFERENCE_ARN")
REGION="us-east-1"

os.makedirs(CACHE_DIR, exist_ok=True)

examples = [
    {
        "intent": "Filtering",
        "question": "Show all products in the electronics category that cost less than $100.",
        "sql": "SELECT * FROM products WHERE category = 'electronics' AND price < 100;"
    },
    {
        "intent": "Aggregation",
        "question": "What is the average price of all products?",
        "sql": "SELECT AVG(price) FROM products;"
    },
    {
        "intent": "Join-related",
        "question": "List names of users and the names of products they ordered.",
        "sql": "SELECT u.name, p.name FROM users u JOIN orders o ON u.user_id = o.user_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id;"
    },
    {
        "intent": "Superlative",
        "question": "Which product has received the highest number of reviews?",
        "sql": "SELECT p.name FROM products p JOIN reviews r ON p.product_id = r.product_id GROUP BY p.product_id, p.name ORDER BY COUNT(*) DESC LIMIT 1;"
    },
    {
        "intent": "Grouping",
        "question": "For each product category, how many products are currently in stock?",
        "sql": "SELECT category, COUNT(*) AS num_products FROM products GROUP BY category;"
    },
    {
        "intent": "Ordering / Ranking",
        "question": "List the top 5 most expensive products.",
        "sql": "SELECT * FROM products ORDER BY price DESC LIMIT 5;"
    },
    {
        "intent": "Boolean Logic",
        "question": "Find premium users who signed up in 2023 or 2024.",
        "sql": "SELECT * FROM users WHERE is_premium = 1 AND YEAR(signup_date) IN (2023, 2024);"
    },
    {
        "intent": "Range Query",
        "question": "Show orders placed between 2023-06-01 and 2023-07-31.",
        "sql": "SELECT * FROM orders WHERE order_date BETWEEN '2023-06-01' AND '2023-07-31';"
    },
    {
        "intent": "Counting",
        "question": "How many orders have been shipped?",
        "sql": "SELECT COUNT(*) AS shipped_orders FROM orders WHERE status = 'shipped';"
    },
    {
        "intent": "Existence Check",
        "question": "Which users have written at least one review?",
        "sql": "SELECT u.* FROM users u WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.user_id = u.user_id);"
    },
    {
        "intent": "Nested / Subquery",
        "question": "List products priced above the average product price.",
        "sql": "SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products);"
    },
    {
        "intent": "Multi-hop Reasoning",
        "question": "Which users purchased electronics products and rated them 5 stars?",
        "sql": "SELECT DISTINCT u.name FROM users u JOIN orders o ON u.user_id = o.user_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id JOIN reviews r ON r.user_id = u.user_id AND r.product_id = p.product_id WHERE p.category = 'electronics' AND r.rating = 5;"
    },
    {
        "intent": "Comparison Between Groups",
        "question": "Which product category has the highest average rating?",
        "sql": "SELECT category FROM (SELECT p.category, AVG(r.rating) AS avg_rating FROM products p JOIN reviews r ON p.product_id = r.product_id GROUP BY p.category) AS cat_ratings ORDER BY avg_rating DESC LIMIT 1;"
    },
    {
        "intent": "Difference / Exclusion",
        "question": "Which products have never been ordered?",
        "sql": "SELECT p.* FROM products p LEFT JOIN order_items oi ON p.product_id = oi.product_id WHERE oi.product_id IS NULL;"
    },
    {
        "intent": "Projection",
        "question": "Show only the name and price of all books.",
        "sql": "SELECT name, price FROM products WHERE category = 'books';"
    },
    {
        "intent": "Lookup / Select",
        "question": "Retrieve full details of the user with user_id = 3.",
        "sql": "SELECT * FROM users WHERE user_id = 3;"
    },
]

example_json = json.dumps(examples, sort_keys=True)
example_hash = hashlib.sha256(example_json.encode()).hexdigest()

if os.path.exists(HASH_FILE) and open(HASH_FILE).read() == example_hash:
    vectorizer = joblib.load(VEC_FILE)
    tfidf_matrix = joblib.load(MAT_FILE)
else:
    questions = [e["question"] for e in examples]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    
    # Cache the results
    with open(HASH_FILE, "w") as f:
        f.write(example_hash)
    joblib.dump(vectorizer, VEC_FILE)
    joblib.dump(tfidf_matrix, MAT_FILE)

def find_top_k_examples(prompt, k=3):
    prompt_vec = vectorizer.transform([prompt])
    similarities = cosine_similarity(prompt_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    return [examples[i] for i in top_indices]

if __name__ == "__main__":
    user_prompt = "List the names of users who ordered a product in the 'electronics' category and gave it a rating below 3."
    k = 3
    results = find_top_k_examples(user_prompt, k)
    
    example_string = ""
    for i, ex in enumerate(results):
        example_string += f"Example {i + 1}\n"
        # example_string += f"Intent: {ex['intent']}\n"
        example_string += f"Question: {ex['question']}\n"
        example_string += f"SQL: {ex['sql']}\n"

    prompt = f"""
You are an expert in converting natural language questions into SQL queries over relational databases.

## Task
Given:
- A natural language question
- A database schema (tables + relationships)
- Example question-SQL pairs

Your job is to generate the correct SQL query to answer the question using the schema and relationships provided.

## Database Schema

Table: users  
- user_id (INT, PK)  
- name (TEXT)  
- email (TEXT)  
- signup_date (DATE)  
- is_premium (BOOLEAN)

Table: products  
- product_id (INT, PK)  
- name (TEXT)  
- category (TEXT)  
- price (DECIMAL)  
- stock (INT)

Table: orders  
- order_id (INT, PK)  
- user_id (INT, FK → users.user_id)  
- order_date (DATE)  
- status (TEXT)

Table: order_items  
- item_id (INT, PK)  
- order_id (INT, FK → orders.order_id)  
- product_id (INT, FK → products.product_id)  
- quantity (INT)  
- price_at_purchase (DECIMAL)

Table: reviews  
- review_id (INT, PK)  
- user_id (INT, FK → users.user_id)  
- product_id (INT, FK → products.product_id)  
- rating (INT)  
- review_text (TEXT)  
- review_date (DATE)

## Example Pairs

{example_string}

## User Question

Question: {user_prompt}

SQL:
"""

    print(prompt)

    client = boto3.client("bedrock-runtime", region_name=REGION)

    response = client.invoke_model(
        modelId=INF_ARN,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.2
        })
    )

    completion = json.loads(response["body"].read())
    print(completion["content"][0]["text"].strip())
