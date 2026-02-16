import sqlite3

def iniciar_db():
    conn = sqlite3.connect('academia.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alunos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            vetor BLOB
        )
    ''')
    conn.commit()
    return conn