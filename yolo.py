# people_counter.py
import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Cargar modelo pre‑entrenado
    model = YOLO("yolov8s.pt")          # o "yolov5s.pt" si prefieres

    # 2. Abrir la cámara (0 = webcam por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    fps_start_time = time.time()
    total_frames = 0
    cv2.namedWindow("YOLO People Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO People Counter", 1920, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1920, 1080))
        # 3. Inferencia
        results = model(frame, conf=0.8)[0]      # Devuelve detecciones

        # 4. Filtrar por clase "person" (id 0 en COCO)
        persons = results.boxes[results.boxes.cls == 0]

        # 5. Dibujar bounding boxes y etiquetas
        for box in persons:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        # 6. Mostrar número de personas en la esquina
        if len(persons) >= 6:
               cv2.putText(frame, f"Ya pueden formar una Cooperativa de {len(persons)} :)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (221, 200, 74), 3)
        else:
            cv2.putText(frame, f"Les falta {6 - len(persons)} personas pa formar una coop bro :'(",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)


        # 7. Mostrar FPS (opcional)
        total_frames += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            fps = total_frames / elapsed
            total_frames = 0
            fps_start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # 8. Mostrar la ventana
        cv2.imshow("YOLO People Counter", frame)

        # 9. Salir con 'q' o ESC
        if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
