import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor

load_dotenv()

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )
    return conn

def fetch_documents():
    """Ambil data dummy dari PostgreSQL untuk dijadikan knowledge base"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    # Contoh query - sesuaikan dengan struktur tabel Anda
    cur.execute("""
        SELECT 
        u.id,
        CONCAT('Profil Siswa: ', u.name) AS title,
        CONCAT(
            'Nama: ', u.name, '. ',
            'Kelas: ', u.grade, '. ',
            'Nomor HP: ', u.phone_number
        ) AS content,
        json_build_object('role', u.role, 'email', u.email, 'parent_id', u.parent_id) AS metadata
        FROM users u
        WHERE u.role = 'student'
        LIMIT 50;
    """)
    
    documents = cur.fetchall()
    cur.close()
    conn.close()
    
    return documents