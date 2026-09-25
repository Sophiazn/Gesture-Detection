"""
Gesture Detection on PC (adaptado do projeto "Gesture Detection on Arduino UNO Q
using Edge Impulse" para rodar 100% local, sem placa Arduino e sem Edge Impulse).

Usa o MediaPipe Tasks GestureRecognizer, que já vem com um modelo pré-treinado
capaz de reconhecer os seguintes gestos de mão:
  Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou

Funciona com qualquer webcam USB comum. Não precisa de conta ou treino no
Edge Impulse — o modelo é baixado automaticamente na primeira execução.

Como rodar:
    pip install -r requirements.txt
    python main.py

Pressione 'q' para sair.
"""

import os
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# Nomes amigáveis para cada gesto reconhecido pelo modelo.
# Edite os valores à vontade para usar as palavras que preferir.
GESTURE_NAMES = {
    "Closed_Fist": "Punho fechado",
    "Open_Palm": "Palma aberta",
    "Pointing_Up": "Apontando",
    "Thumb_Down": "Ruim",
    "Thumb_Up": "Bom",
    "Victory": "Paz",
    "ILoveYou": "Eu te amo",
    "None": "Nenhum gesto",
}


def friendly_name(category_name: str) -> str:
    """Traduz o rótulo técnico do MediaPipe para o nome amigável configurado acima."""
    return GESTURE_NAMES.get(category_name, category_name)


def ensure_model() -> None:
    """Baixa o modelo pré-treinado do MediaPipe caso ainda não exista localmente."""
    if os.path.exists(MODEL_PATH):
        return
    print("Baixando modelo de reconhecimento de gestos (uma única vez)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"Falha ao baixar o modelo automaticamente: {exc}")
        print(f"Baixe manualmente em:\n  {MODEL_URL}")
        print(f"E salve como:\n  {MODEL_PATH}")
        sys.exit(1)
    print("Modelo baixado com sucesso.")


def list_cameras(max_index: int = 5) -> list[int]:
    """Testa os índices de câmera disponíveis (0 a max_index) e retorna os que abrem."""
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def choose_camera() -> int:
    """Deixa o usuário escolher qual câmera usar, caso haja mais de uma."""
    cameras = list_cameras()
    if not cameras:
        print("Nenhuma câmera encontrada.")
        sys.exit(1)
    if len(cameras) == 1:
        return cameras[0]

    print("Câmeras encontradas:")
    for idx in cameras:
        print(f"  [{idx}] Câmera {idx}")
    while True:
        choice = input(f"Digite o número da câmera que deseja usar {cameras}: ").strip()
        if choice.isdigit() and int(choice) in cameras:
            return int(choice)
        print("Opção inválida, tente de novo.")


def main() -> None:
    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    camera_index = choose_camera()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Não foi possível abrir a webcam (índice {camera_index}). Verifique a conexão/permissões.")
        sys.exit(1)

    print("Câmera aberta. Pressione 'q' na janela de vídeo para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao capturar frame da webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = recognizer.recognize(mp_image)

        h, w, _ = frame.shape

        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                xs = [lm.x * w for lm in hand_landmarks]
                ys = [lm.y * h for lm in hand_landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                # Caixa delimitadora ao redor da mão (equivalente ao bounding box do demo original)
                cv2.rectangle(
                    frame,
                    (max(0, x_min - 20), max(0, y_min - 20)),
                    (min(w, x_max + 20), min(h, y_max + 20)),
                    (0, 255, 0),
                    2,
                )

                # Rótulo do gesto reconhecido
                if idx < len(result.gestures) and result.gestures[idx]:
                    gesture = result.gestures[idx][0]
                    label = f"{friendly_name(gesture.category_name)} ({gesture.score:.2f})"
                    cv2.putText(
                        frame,
                        label,
                        (max(0, x_min - 20), max(20, y_min - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Pontos dos landmarks da mão
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        cv2.imshow("Gesture Detection (PC)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
