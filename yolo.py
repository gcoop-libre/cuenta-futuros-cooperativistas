# people_counter_improved.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque


class PeopleCounter:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.cap = cv2.VideoCapture(0)
        
        # Logo
        self.logo = None
        self.logo_path = "./logo-horizonta.png"  # Cambia esto por la ruta de tu imagen
        self.logo_width = 200  # Ancho del logo en píxeles
        self.load_logo()

        # Colores modernos (BGR)
        self.COLOR_PRIMARY = (255, 107, 53)      # Naranja moderno
        self.COLOR_SUCCESS = (98, 196, 98)       # Verde suave
        self.COLOR_WARNING = (71, 130, 255)      # Naranja-rojo
        self.COLOR_ACCENT = (255, 215, 64)       # Amarillo dorado
        self.COLOR_DARK = (30, 30, 30)           # Gris oscuro
        self.COLOR_LIGHT = (240, 240, 240)       # Gris claro
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.fps_start_time = time.time()
        self.total_frames = 0
        
        # Animación del contador
        self.current_count = 0
        self.target_count = 0
        
        # Para countdown y captura de frame
        self.countdown_start_time = None
        self.success_time = None
        self.frozen_frame = None
        self.frozen_count = None  # Guarda el número de personas cuando inicia countdown
        
        # Configuración de ventana
        self.window_name = "Contador de cooperativistas"
        
    def create_gradient_overlay(self, frame, alpha=0.3):
        """Crea un overlay con gradiente para el header"""
        overlay = frame.copy()
        height = 120
        
        # Gradiente de negro a transparente
        for i in range(height):
            alpha_val = int(255 * (1 - i / height) * alpha)
            overlay[i, :] = cv2.addWeighted(overlay[i, :], 1, 
                                           np.full_like(overlay[i, :], 0), 
                                           alpha_val / 255, 0)
        
        return overlay
    
    def load_logo(self):
        """Carga y redimensiona el logo"""
        try:
            logo_img = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                # Calcular altura manteniendo proporción
                aspect_ratio = logo_img.shape[0] / logo_img.shape[1]
                logo_height = int(self.logo_width * aspect_ratio)

                # Redimensionar
                self.logo = cv2.resize(logo_img, (self.logo_width, logo_height))
                print(f"Logo cargado: {self.logo_path} ({self.logo_width}x{logo_height})")
            else:
                print(f"Advertencia: No se pudo cargar el logo desde {self.logo_path}")
        except Exception as e:
            print(f"Error al cargar logo: {e}")

    def draw_logo(self, frame):
        """Dibuja el logo en la esquina superior derecha"""
        if self.logo is None:
            return

        # Posición: esquina superior derecha con padding
        padding = 20
        y_offset = padding
        x_offset = frame.shape[1] - self.logo.shape[1] - padding

        # Si el logo tiene canal alpha (transparencia)
        if self.logo.shape[2] == 4:
            # Extraer el canal alpha
            alpha = self.logo[:, :, 3] / 255.0

            # Región donde se va a poner el logo
            y1, y2 = y_offset, y_offset + self.logo.shape[0]
            x1, x2 = x_offset, x_offset + self.logo.shape[1]

            # Mezclar con transparencia
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * self.logo[:, :, c] +
                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            # Sin transparencia, pegar directo
            y1, y2 = y_offset, y_offset + self.logo.shape[0]
            x1, x2 = x_offset, x_offset + self.logo.shape[1]
            frame[y1:y2, x1:x2] = self.logo[:, :, :3]

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius=15):
        """Dibuja un rectángulo con esquinas redondeadas"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Líneas horizontales
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        
        # Líneas verticales
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Esquinas redondeadas
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_filled_rounded_rectangle(self, img, pt1, pt2, color, radius=15):
        """Dibuja un rectángulo relleno con esquinas redondeadas"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Rectángulos para el cuerpo
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Círculos para las esquinas
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    
    def draw_person_box(self, frame, box, index):
        """Dibuja una caja estilizada para cada persona detectada"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Color degradado
        color_intensity = int(conf * 255)
        box_color = (color_intensity // 2, color_intensity, color_intensity // 2)
        
        # Dibuja rectángulo principal con esquinas redondeadas
        self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), box_color, 3, radius=10)
        
        # Esquinas decorativas
        corner_length = 20
        corner_thickness = 4
        
        # Esquina superior izquierda
        cv2.line(frame, (x1, y1 + corner_length), (x1, y1), box_color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, corner_thickness)
        
        # Esquina superior derecha
        cv2.line(frame, (x2 - corner_length, y1), (x2, y1), box_color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), box_color, corner_thickness)
        
        # Esquina inferior izquierda
        cv2.line(frame, (x1, y2 - corner_length), (x1, y2), box_color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), box_color, corner_thickness)
        
        # Esquina inferior derecha
        cv2.line(frame, (x2 - corner_length, y2), (x2, y2), box_color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, corner_thickness)
        
        # Etiqueta con fondo
        label = f"Cooperativista #{index + 1}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Fondo de la etiqueta
        self.draw_filled_rounded_rectangle(
            frame, 
            (x1, y1 - label_h - 20), 
            (x1 + label_w + 20, y1),
            box_color,
            radius=8
        )
        
        # Texto de la etiqueta
        cv2.putText(frame, label, (x1 + 10, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        conf_text = f"{conf:.0%}"
        cv2.circle(frame, (x2 - 30, y1 + 30), 25, self.COLOR_DARK, -1)
        cv2.circle(frame, (x2 - 30, y1 + 30), 23, box_color, 2)
        
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, conf_text, 
                    (x2 - 30 - conf_w // 2, y1 + 30 + conf_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_countdown(self, frame, countdown_value):
        """Dibuja el contador regresivo en el centro de la pantalla"""
        height, width = frame.shape[:2]
        
        # Fondo semi-transparente SOLO en el área del texto
        overlay = frame.copy()
        
        # Texto del countdown
        if countdown_value > 0:
            countdown_text = str(countdown_value)
            font_scale = 15
            thickness = 18
            color = self.COLOR_ACCENT
            msg = "Preparandose para la foto..."
        else:
            countdown_text = "A COOPERAR!"
            font_scale = 3
            thickness = 8
            color = self.COLOR_SUCCESS
            msg = ""
        
        # Calcular tamaño y posición
        (text_w, text_h), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x = (width - text_w) // 2
        text_y = (height + text_h) // 2
        
        # Crear rectángulo oscuro solo detrás del número
        padding = 50
        cv2.rectangle(overlay, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Sombra
        cv2.putText(frame, countdown_text, (text_x + 5, text_y + 5),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 5)
        
        # Texto principal
        cv2.putText(frame, countdown_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        
        # Mensaje adicional
        if msg:
            (msg_w, msg_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            msg_x = (width - msg_w) // 2
            msg_y = text_y + 100
            
            # Fondo del mensaje
            cv2.rectangle(overlay, 
                         (msg_x - 20, msg_y - msg_h - 10),
                         (msg_x + msg_w + 20, msg_y + 10),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, msg, (msg_x, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_LIGHT, 3)

    def draw_header(self, frame, person_count, fps):
        """Dibuja header"""
        height, width = frame.shape[:2]

        # Fondo del header con transparencia
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), self.COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Espacio reservado para logo
        logo_space = 250

        # Título centrado
        title_text = "Cuenta futuros cooperativistas"
        (title_w, title_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        available_width = width - 2 * logo_space
        title_x = logo_space + (available_width - title_w) // 2
        cv2.putText(frame, title_text, (title_x, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_LIGHT, 2)

        # Línea decorativa
        cv2.line(frame, (logo_space, 55), (width - logo_space, 55), self.COLOR_PRIMARY, 2)

        # Indicador de estado
        status_text = "SISTEMA ACTIVO"
        cv2.circle(frame, (logo_space + 10, 75), 8, self.COLOR_SUCCESS, -1)
        cv2.putText(frame, status_text, (logo_space + 25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_LIGHT, 1)

        # FPS en el header
        fps_text = f"FPS: {fps:.1f}"
        (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, fps_text, (width - logo_space - fps_w - 10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ACCENT, 1)
    
    def draw_counter_panel(self, frame, person_count, is_success_mode):
        """Dibuja un panel lateral con el contador principal"""
        height, width = frame.shape[:2]
        
        # Panel lateral derecho
        panel_width = 300
        panel_x = width - panel_width - 20
        panel_y = 120
        panel_height = 200
        
        # Fondo del panel
        overlay = frame.copy()
        self.draw_filled_rounded_rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.COLOR_DARK,
            radius=20
        )
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Borde del panel
        self.draw_rounded_rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.COLOR_PRIMARY,
            3,
            radius=20
        )
        
        # Título del panel - CAMBIA según si llegó a 6 o no
        if is_success_mode:
            title_panel = "COOPERATIVISTAS"
        else:
            title_panel = "POSIBLES COOPERATIVISTAS"
        
        (title_pw, _), _ = cv2.getTextSize(title_panel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        title_px = panel_x + (panel_width - title_pw) // 2
        cv2.putText(frame, title_panel, 
                    (title_px, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_LIGHT, 2)

        # Contador grande - CONGELADO en el número cuando inició countdown
        if is_success_mode and self.frozen_count is not None:
            display_count = self.frozen_count
        else:
            self.target_count = person_count
            if self.current_count < self.target_count:
                self.current_count = min(self.current_count + 1, self.target_count)
            elif self.current_count > self.target_count:
                self.current_count = max(self.current_count - 1, self.target_count)
            display_count = int(self.current_count)
        
        count_text = str(display_count)
        (count_w, count_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 4, 6)
        count_x = panel_x + (panel_width - count_w) // 2
        count_y = panel_y + 150
        
        # Sombra del número
        cv2.putText(frame, count_text, (count_x + 3, count_y + 3),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 6)
        
        # Número principal
        number_color = self.COLOR_SUCCESS if is_success_mode or person_count >= 6 else self.COLOR_WARNING
        cv2.putText(frame, count_text, (count_x, count_y),
                    cv2.FONT_HERSHEY_DUPLEX, 4, number_color, 6)
    
    def draw_cooperative_status(self, frame, person_count, force_success=False):
        """Dibuja cuadro de cooperativistas que faltan"""
        height, width = frame.shape[:2]
        
        panel_width = 950
        panel_height = 100
        panel_x = (width - panel_width) // 2
        panel_y = height - panel_height - 30
        
        # Fondo del panel
        overlay = frame.copy()
        
        if person_count >= 6 or force_success:
            # Panel de éxito
            self.draw_filled_rounded_rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                self.COLOR_SUCCESS,
                radius=20
            )
            message = "PUEDEN FORMAR UNA COOPERATIVA!"
            text_color = (255, 255, 255)
            border_color = (255, 255, 255)
        else:
            # Panel de progreso
            self.draw_filled_rounded_rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                self.COLOR_DARK,
                radius=20
            )
            faltantes = 6 - person_count
            message = f"Les faltan {faltantes} cooperativista{'s' if faltantes > 1 else ''} para formar la cooperativa"
            text_color = (255, 255, 255)
            border_color = self.COLOR_PRIMARY
        
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Borde
        self.draw_rounded_rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            border_color,
            3,
            radius=20
        )
        
        # Texto del mensaje
        (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        text_x = panel_x + (panel_width - text_w) // 2
        text_y = panel_y + (panel_height + text_h) // 2
        
        # Sombra
        cv2.putText(frame, message, (text_x + 2, text_y + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3)
        
        # Texto principal
        cv2.putText(frame, message, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color, 2)
        
        # NO mostrar barra de progreso durante éxito
        if person_count < 6 and not force_success:
            bar_width = panel_width - 80
            bar_height = 12
            bar_x = panel_x + 40
            bar_y = panel_y + 70
            
            # Fondo de la barra
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Progreso
            progress = person_count / 6
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         self.COLOR_PRIMARY, -1)
    
    def run(self):
        """Ejecuta el contador de cooperativistas"""
        if not self.cap.isOpened():
            print("Error: no se pudo abrir la camara.")
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        print("Sistema iniciado. Presiona 'Q' o 'ESC' para salir.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Redimensionar frame
            frame = cv2.resize(frame, (1280, 720))
            
            # Inferencia YOLO
            results = self.model(frame, conf=0.5, verbose=False)[0]
            
            # Filtrar personas
            persons = results.boxes[results.boxes.cls == 0]
            person_count = len(persons)
            
            # Dibuja cajas de personas
            for idx, box in enumerate(persons):
                self.draw_person_box(frame, box, idx)
            
            # Calcular FPS
            self.total_frames += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0.5:
                fps = self.total_frames / elapsed
                self.fps_history.append(fps)
                self.total_frames = 0
                self.fps_start_time = time.time()
            
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            
            # Determinar si estamos en modo éxito
            is_success_mode = (self.countdown_start_time is not None) or (self.frozen_frame is not None)
            
            # SIEMPRE dibuja la interfaz base
            self.draw_header(frame, person_count, avg_fps)
            self.draw_logo(frame)
            self.draw_counter_panel(frame, person_count, is_success_mode)
            self.draw_cooperative_status(frame, person_count, force_success=is_success_mode)
            
            # INICIAR COUNTDOWN CUANDO LLEGA A 6
            if person_count >= 6 and self.countdown_start_time is None and self.frozen_frame is None:
                self.countdown_start_time = time.time()
                self.frozen_count = person_count  # GUARDAR el número actual de personas
                print(f"Iniciando cuenta regresiva: 5... 4... 3... 2... 1... ({self.frozen_count} personas detectadas)")
            
            # MOSTRAR COUNTDOWN Y CAPTURAR FRAME
            if self.countdown_start_time is not None and self.frozen_frame is None:
                elapsed_countdown = time.time() - self.countdown_start_time
                countdown_value = 5 - int(elapsed_countdown)
                
                if countdown_value >= 0:
                    # Dibujar countdown ENCIMA de la interfaz de éxito
                    self.draw_countdown(frame, countdown_value)
                    cv2.imshow(self.window_name, frame)
                else:
                    # CAPTURAR FRAME cuando llega a 0
                    self.success_time = time.time()
                    self.frozen_frame = frame.copy()
                    self.countdown_start_time = None
                    print("Foto capturada! Mostrando por 15 segundos...")
            
            # MOSTRAR FRAME CONGELADO O NORMAL
            elif self.frozen_frame is not None and (time.time() - self.success_time) < 15:
                # Mostrar la fotografía congelada
                cv2.imshow(self.window_name, self.frozen_frame)
            elif self.frozen_frame is not None:
                # Después de 15 segundos, resetear todo
                print("Reseteando contador...")
                self.frozen_frame = None
                self.success_time = None
                self.countdown_start_time = None
                self.frozen_count = None  # RESETEAR el contador congelado
                self.current_count = 0
                self.target_count = 0
                cv2.imshow(self.window_name, frame)
            else:
                # Mostrar video en vivo normal
                cv2.imshow(self.window_name, frame)
            
            # Salir con 'q' o ESC
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), 27}:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sistema detenido correctamente.")


def main():
    counter = PeopleCounter()
    counter.run()


if __name__ == "__main__":
    main()