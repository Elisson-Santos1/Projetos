import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from database import iniciar_db

TARGET_FPS = 60
IA_RES = (320, 320)
TEMPO_VALIDACAO = 3.0 

class Portaria:
    def __init__(self):
        self.conn = iniciar_db()
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=IA_RES)
        self.carregar_cache()
        
        self.aluno_atual = None
        self.inicio_detect = 0

    def carregar_cache(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT nome, vetor FROM alunos")
        registros = cursor.fetchall()
        self.nomes = [r[0] for r in registros]
        self.vetores = np.array([np.frombuffer(r[1], dtype=np.float32) for r in registros]) if registros else []

    def verificar(self):
        webcam = cv2.VideoCapture(0)
        print(f"🚀 Aguardando validação de {TEMPO_VALIDACAO}s...")

        while True:
            ret, frame = webcam.read()
            if not ret: break

            faces = self.app.get(frame)
            face_detectada_neste_frame = False

            for face in faces:
                nome_detectado = "DESCONHECIDO"
                cor = (0, 0, 255)
                
                if len(self.vetores) > 0:
                    sims = np.dot(self.vetores, face.normed_embedding)
                    idx = np.argmax(sims)
                    
                    if sims[idx] > 0.50:
                        nome_detectado = self.nomes[idx]
                        cor = (0, 255, 255) 
                        face_detectada_neste_frame = True

                if face_detectada_neste_frame:
                    if self.aluno_atual == nome_detectado:
                        tempo_passado = time.time() - self.inicio_detect
                        
                        progresso = min(tempo_passado / TEMPO_VALIDACAO, 1.0)
                        b = face.bbox.astype(int)
                        cv2.rectangle(frame, (b[0], b[3] + 5), (b[0] + int((b[2]-b[0]) * progresso), b[3] + 15), (0, 255, 0), -1)
                        
                        if tempo_passado >= TEMPO_VALIDACAO:
                            webcam.release()
                            cv2.destroyAllWindows()
                            return nome_detectado
                    else:
                        self.aluno_atual = nome_detectado
                        self.inicio_detect = time.time()
                
                b = face.bbox.astype(int)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), cor, 2)
                cv2.putText(frame, nome_detectado, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            if not faces:
                self.aluno_atual = None

            cv2.imshow("PORTARIA - FIQUE PARADO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        webcam.release()
        cv2.destroyAllWindows()
        return None

if __name__ == "__main__":
    p = Portaria()
    resultado = p.verificar()
    if resultado:
        print(f"\n✅ ACESSO LIBERADO: {resultado}\n👉 PODE PASSAR!")