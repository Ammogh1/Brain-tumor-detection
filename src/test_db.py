import sys
sys.path.append('app')
from database import get_connection

conn, err = get_connection()
if conn:
    print("SUCCESS: Connection established!")
else:
    print(f"FAILED: {err}")
