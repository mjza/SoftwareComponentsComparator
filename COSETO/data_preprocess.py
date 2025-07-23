import re
import os
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL) 

def pre_process(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    return " ".join(tokens)

# Step 1: Read all issue_id and issue_text values
with engine.connect() as conn:
    df = pd.read_sql("SELECT issue_id, issue_text FROM combined_issues", conn)

# Step 2: Apply preprocessing
df['clean_issue_text'] = df['issue_text'].apply(pre_process)

# Step 3: Write back clean_issue_text to DB
with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(
            text("UPDATE combined_issues SET clean_issue_text = :clean_text WHERE issue_id = :issue_id"),
            {"clean_text": row['clean_issue_text'], "issue_id": row['issue_id']}
        )
