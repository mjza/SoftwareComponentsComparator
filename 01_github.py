import os
import sqlite3
import requests
import time
import json
import psycopg2
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BATCH_SIZE = 10
REQUEST_COUNTER = 0
MAX_REQUESTS = 4999


# returns a db connection
def get_connection(DBMS):
    if DBMS == 'SQLITE':
        conn = sqlite3.connect(os.getenv('DB_PATH'))
    elif DBMS == 'POSTGRES':
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
    else:
        raise ValueError("Unsupported DBMS")
    return conn


# Fetch rate limits from GitHub API
def get_rate_limits():
    rate_limit_url = "https://api.github.com/rate_limit"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(rate_limit_url, headers=headers)
    if response and response.status_code == 200:
        return response.json()['resources']['core']
    return None


# Safe request with rate limit handling
def safe_request(url, headers, params=None, max_retries=3, delay=5):
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    
    if REQUEST_COUNTER >= MAX_REQUESTS:
        rate_limits = get_rate_limits()
        if rate_limits and rate_limits['remaining'] == 0:
            reset_time = rate_limits['reset']
            sleep_duration = max(reset_time - time.time(), 1)
            print(f"Rate limit reached. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)
            REQUEST_COUNTER = 0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Attempt {attempt + 1} of {max_retries}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print("All retry attempts failed.")
    return None


# Create issues and comments tables if not exists
def create_tables():
    conn = get_connection('POSTGRES')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            issue_id BIGINT PRIMARY KEY,
            url TEXT,
            project_id BIGINT,
            repository_id BIGINT,
            repository_url TEXT,
            node_id TEXT,
            number INTEGER,
            title TEXT,
            owner TEXT,
            owner_type TEXT,
            owner_id BIGINT,
            labels TEXT,
            state TEXT,
            locked BOOLEAN,
            comments INTEGER,
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE,
            closed_at TIMESTAMP WITH TIME ZONE,
            author_association TEXT,
            active_lock_reason TEXT,
            body TEXT,
            body_text TEXT,
            reactions TEXT,
            state_reason TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id BIGINT PRIMARY KEY,
            node_id TEXT,
            url TEXT,
            issue_id BIGINT,
            issue_url TEXT,
            owner TEXT,
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE,
            author_association TEXT,
            body TEXT,
            body_text TEXT,
            FOREIGN KEY(issue_id) REFERENCES issues(issue_id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()


# Fetch project repositories in batches
def fetch_projects():
    conn = get_connection('SQLITE')
    cursor = conn.cursor()
    offset = 0
    
    while True:
        cursor.execute("SELECT id, repository_url FROM projects LIMIT ? OFFSET ?", (BATCH_SIZE, offset))
        projects = cursor.fetchall()
        if not projects:
            break
        yield projects
        offset += BATCH_SIZE
    
    conn.close()


# Fetch repository ID from GitHub
def fetch_repository_id(repo_url):
    repo_path = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{repo_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}",
               'User-Agent': 'Mozilla/5.0',
               'Accept': 'application/vnd.github+json'}
    response = safe_request(api_url, headers=headers)
    if response and response.status_code == 200:
        return response.json().get("id")
    print(f"Failed to fetch repository ID for {repo_url}: {response.status_code}")
    return None


# Fetch issues from GitHub
def fetch_github_issues(repo_url, project_id, repository_id):
    if not repo_url or "github.com" not in repo_url:
        return
    
    conn = get_connection('POSTGRES')
    repo_path = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{repo_path}/issues"
    headers = {"Authorization": f"token {GITHUB_TOKEN}",
               'User-Agent': 'Mozilla/5.0',
               'Accept': 'application/vnd.github.full+json'}
    params = {
                'state': 'all',
                'sort': 'created',
                'direction': 'asc',
                'per_page': 100,
                'page': 1
            }
        
    has_more_pages = True
    while has_more_pages:
        try:
            response = safe_request(api_url, headers=headers, params=params)
            if response and response.status_code == 200:
                issues = response.json()
                if not issues:
                    break

                for issue in issues:
                    try:
                        issue['project_id'] = project_id
                        issue['repository_id'] = repository_id
                        issue_url = issue.get('url')
                        insert_issue_data(conn, issue)
                        fetch_github_comments(issue_url, issue.get('id', 0))
                    except Exception as e:
                        print(f"Error processing issue {issue.get('id', 'Unknown')} for project {project_id}: {e}")
                        exit(1)
                print(f"Stored {len(issues)} issues for repository {repo_url}")
                if 'next' in response.links:
                    params['page'] += 1
                else:
                    has_more_pages = False
            elif response and response.status_code >= 400:
                print(f"Failed to fetch issues for repository {api_url}. HTTP {response.status_code}, Error: {response.text}")
                has_more_pages = False
            else:
                print(f"Failed to fetch issues for repository {api_url}. No response is available")
                has_more_pages = False
        except Exception as e:
            print(f"Exception occurred while fetching issues for project {project_id}: {e}")
            has_more_pages = False  # Prevent further attempts to fetch pages
    
    conn.close()


# Fetch comments from GitHub
def fetch_github_comments(issue_url, issue_id):
    if not issue_url or "github.com" not in issue_url:
        return
    
    conn = get_connection('POSTGRES')
    api_url = f"{issue_url}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}",
               'User-Agent': 'Mozilla/5.0',
               'Accept': 'application/vnd.github.full+json'}
    params = {
                'sort': 'created',
                'direction': 'asc',
                'per_page': 100,
                'page': 1
            }
        
    has_more_pages = True
    while has_more_pages:
        try:
            response = safe_request(api_url, headers=headers, params=params)
            if response and response.status_code == 200:
                comments = response.json()
                if not comments:
                    break

                for comment in comments:
                    try:
                        comment['issue_id'] = issue_id
                        insert_comment_data(conn, comment)
                    except Exception as e:
                        print(f"Error processing comment {comment.get('id', 'Unknown')} for issue {issue_id}: {e}")
                        exit(1)
                if 'next' in response.links:
                    params['page'] += 1
                else:
                    has_more_pages = False
            elif response and response.status_code >= 400:
                print(f"Failed to fetch comments for repository {api_url}. HTTP {response.status_code}, Error: {response.text}")
                has_more_pages = False
            else:
                print(f"Failed to fetch comments for repository {api_url}. No response is available")
                has_more_pages = False
        except Exception as e:
            print(f"Exception occurred while fetching comments for issue {issue_id}: {e}")
            has_more_pages = False  # Prevent further attempts to fetch pages
            
    conn.close()
  


# Insert issue data into database
def insert_issue_data(conn, issue_data):
    cursor = conn.cursor()
    sql = """
    INSERT INTO issues 
    (issue_id, url, project_id, repository_id, repository_url, node_id, number, title, owner, owner_type, owner_id, labels, state, locked, comments, created_at, updated_at, closed_at, author_association, active_lock_reason, body, body_text, reactions, state_reason) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(issue_id) DO UPDATE SET 
        url = EXCLUDED.url,
        project_id = EXCLUDED.project_id,
        repository_id = EXCLUDED.repository_id,
        repository_url = EXCLUDED.repository_url,
        node_id = EXCLUDED.node_id,
        number = EXCLUDED.number,
        title = EXCLUDED.title,
        owner = EXCLUDED.owner,
        owner_type = EXCLUDED.owner_type,
        owner_id = EXCLUDED.owner_id,
        labels = EXCLUDED.labels,
        state = EXCLUDED.state,
        locked = EXCLUDED.locked,        
        comments = EXCLUDED.comments,
        created_at = EXCLUDED.created_at,
        updated_at = EXCLUDED.updated_at,
        closed_at = EXCLUDED.closed_at,
        author_association = EXCLUDED.author_association,
        active_lock_reason = EXCLUDED.active_lock_reason,
        body = EXCLUDED.body,
        body_text = EXCLUDED.body_text,
        reactions = EXCLUDED.reactions,
        state_reason = EXCLUDED.state_reason
    """
    cursor.execute(sql, (
        issue_data.get('id'), issue_data.get('url'), issue_data.get('project_id'), issue_data.get('repository_id'), issue_data.get('repository_url'), 
        issue_data.get('node_id'), issue_data.get('number'), issue_data.get('title'), issue_data.get('user', {}).get('login'), 
        issue_data.get('user', {}).get('type'), issue_data.get('user', {}).get('id'),
        json.dumps(issue_data.get('labels', [])), issue_data.get('state'), issue_data.get('locked'), 
        issue_data.get('comments'), issue_data.get('created_at'), 
        issue_data.get('updated_at'), issue_data.get('closed_at'), issue_data.get('author_association'), 
        issue_data.get('active_lock_reason'), issue_data.get('body'), issue_data.get('body_text'), json.dumps(issue_data.get('reactions', {})), issue_data.get('state_reason')
    ))
    conn.commit()


# Insert comment data into database
def insert_comment_data(conn, comment_data):
    cursor = conn.cursor()
    sql = """
    INSERT INTO comments 
    (id, node_id, url, issue_id, issue_url, owner, created_at, updated_at, author_association, body, body_text) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(id) DO UPDATE SET 
        node_id = EXCLUDED.node_id,
        url = EXCLUDED.url,
        issue_id = EXCLUDED.issue_id,
        issue_url = EXCLUDED.issue_url,
        owner = EXCLUDED.owner,
        created_at = EXCLUDED.created_at,
        updated_at = EXCLUDED.updated_at,
        author_association = EXCLUDED.author_association,
        body = EXCLUDED.body,
        body_text = EXCLUDED.body_text
    """
    cursor.execute(sql, (
        comment_data.get('id'), comment_data.get('node_id'), comment_data.get('url'),
        comment_data.get('issue_id'), comment_data.get('issue_url'),
        comment_data.get('user', {}).get('login'), comment_data.get('created_at'),
        comment_data.get('updated_at'), comment_data.get('author_association'),
        comment_data.get('body'), comment_data.get('body_text')
    ))
    conn.commit()
    

# Main execution
def main():
    create_tables()
    
    for batch in fetch_projects():
        for project_id, repo_url in batch:
            repository_id = fetch_repository_id(repo_url)
            if repository_id:
                fetch_github_issues(repo_url, project_id, repository_id)

    
if __name__ == "__main__":
    main()
