import cv2
import numpy as np
import time
import os
from insightface.app import FaceAnalysis
from database import iniciar_db

if not os.path.exists('fotos_alunos'):
    os.makedirs('fotos_alunos')

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def cadastrar():
    conn = iniciar_db()
    webcam = cv2.VideoCapture(0)
    print("📸 MODO CADASTRO")
    print("Aperte 'S' para capturar o frame atual e salvar.")

    while True:
        ret, frame = webcam.read()
        if not ret: break
        
        exibicao = frame.copy()
        faces = app.get(frame)
        
        for face in faces:
            b = face.bbox.astype(int)
            cv2.rectangle(exibicao, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

        cv2.imshow("CADASTRO - PRESSIONE 'S' PARA SALVAR", exibicao)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            nome = input("Digite o nome do aluno: ")
            face = faces[0]
            vetor = face.normed_embedding
            
            foto_path = f"fotos_alunos/{nome.replace(' ', '_')}_{int(time.time())}.jpg"
            cv2.imwrite(foto_path, frame) 
            
            try:
                conn.execute("ALTER TABLE alunos ADD COLUMN foto_path TEXT")
            except:
                pass
                
            conn.execute("INSERT INTO alunos (nome, vetor, foto_path) VALUES (?, ?, ?)", 
                          (nome, vetor.tobytes(), foto_path))
            conn.commit()
            
            print(f"✅ Sucesso! Aluno: {nome}")
            print(f"🖼️ Imagem salva em: {foto_path}")
            break
            
        elif key == ord('q'): 
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cadastrar()