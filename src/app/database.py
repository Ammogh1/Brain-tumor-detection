import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "braintumor_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn, None
    except Exception as e:
        return None, str(e)

def init_db():
    """Initializes the required tables inside the Postgres schema."""
    conn, err = get_connection()
    if conn is None: return err
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    image BYTEA NOT NULL,
                    predicted_class VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        return "Success"
    except Exception as e:
        return str(e)
    finally:
        conn.close()

def insert_prediction(image_bytes, predicted_class, confidence):
    conn, err = get_connection()
    if conn is None: return err
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (image, predicted_class, confidence)
                VALUES (%s, %s, %s);
            """, (psycopg2.Binary(image_bytes), predicted_class, confidence))
        conn.commit()
        return "Success"
    except Exception as e:
        return str(e)
    finally:
        conn.close()

def get_recent_predictions(limit=10):
    conn, err = get_connection()
    if conn is None: return [] # Fails silently for sidebar view
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s;
            """, (limit,))
            rows = cur.fetchall()
            return rows
    except Exception as e:
        return []
    finally:
        if conn is not None:
            conn.close()
