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

def fetch_teacher_documents(user_id: str) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute("""
        SELECT 
            u.id AS teacher_id,
            u.name AS teacher_name,
            COUNT(DISTINCT st.student_id) AS student_count,
            COUNT(DISTINCT qp.id) AS package_count
        FROM users u
        LEFT JOIN student_teacher st ON st.teacher_id = u.id
        LEFT JOIN question_packages qp ON qp.created_by = u.id
        WHERE u.role = 'teacher' AND u.id = %s
        GROUP BY u.id, u.name
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    documents = []
    for row in rows:
        content = (
            f"Guru {row['teacher_name']} membimbing {row['student_count']} siswa "
            f"dan telah membuat {row['package_count']} paket soal."
        )
        documents.append({
            "id": str(row["teacher_id"]),
            "title": f"Data Guru {row['teacher_name']}",
            "content": content,
            "metadata": {
                "user_role": "teacher",
                "teacher_id": str(row["teacher_id"])
            }
        })
    return documents

def fetch_student_documents(user_id: str) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute("""
        SELECT 
            s.id AS student_id,
            s.name AS student_name,
            p.name AS parent_name,
            COUNT(DISTINCT a.id) AS total_answers,
            AVG(qr.score) AS avg_score,
            lp.preferred_learning_style,
            lp.learning_pace
        FROM users s
        LEFT JOIN users p ON s.parent_id = p.id
        LEFT JOIN answers a ON a.student_id = s.id
        LEFT JOIN quiz_results qr ON qr.student_id = s.id
        LEFT JOIN learning_profiles lp ON lp.student_id = s.id
        WHERE s.role = 'student' AND s.id = %s
        GROUP BY s.id, s.name, p.name, lp.preferred_learning_style, lp.learning_pace
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    documents = []
    for row in rows:
        content = (
            f"Nama siswa: {row['student_name']}. "
            f"Orang tua: {row['parent_name'] or 'Belum terdaftar'}. "
            f"Telah menjawab {row['total_answers']} soal, rata-rata skor: {row['avg_score'] or 0:.2f}. "
            f"Gaya belajar: {row['preferred_learning_style'] or 'Tidak diketahui'}, "
            f"kecepatan belajar: {row['learning_pace'] or 'Tidak diketahui'}."
        )
        documents.append({
            "id": str(row["student_id"]),
            "title": f"Profil Siswa {row['student_name']}",
            "content": content,
            "metadata": {
                "user_role": "student",
                "student_id": str(row["student_id"])
            }
        })
    return documents

def fetch_parent_documents(user_id: str) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute("""
        SELECT 
            p.id AS parent_id,
            p.name AS parent_name,
            ARRAY_AGG(s.name) FILTER (WHERE s.name IS NOT NULL) AS children_names,
            AVG(qr.score) AS avg_children_score
        FROM users p
        LEFT JOIN users s ON s.parent_id = p.id
        LEFT JOIN quiz_results qr ON qr.student_id = s.id
        WHERE p.role = 'parent' AND p.id = %s
        GROUP BY p.id, p.name
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    documents = []
    for row in rows:
        anak = ', '.join(row['children_names']) if row['children_names'] else "Belum ada anak terdaftar"
        content = (
            f"Orang tua: {row['parent_name']}.\n"
            f"Anak-anak: {anak}.\n"
            f"Rata-rata skor anak: {row['avg_children_score'] or 0:.2f}."
        )
        documents.append({
            "id": str(row["parent_id"]),
            "title": f"Data Orang Tua {row['parent_name']}",
            "content": content,
            "metadata": {
                "user_role": "parent",
                "parent_id": str(row["parent_id"])
            }
        })
    return documents


def fetch_documents_by_role(user_id: str, role: str) -> list[dict]:
    if role == "teacher":
        return fetch_teacher_documents(user_id)
    elif role == "student":
        return fetch_student_documents(user_id)
    elif role == "parent":
        return fetch_parent_documents(user_id)
    else:
        raise ValueError("Invalid role")
