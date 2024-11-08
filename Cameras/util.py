import cv2
import os
from pathlib import Path

def extract_images_from_video(video_path, output_path, start_time, end_time, interval):
    """
    Extrae imágenes de un video en intervalos de tiempo específicos dentro de un rango determinado.

    :param video_path: Ruta al archivo de video.
    :param output_path: Ruta a la carpeta donde se guardarán las imágenes extraídas.
    :param start_time: Tiempo de inicio en segundos.
    :param end_time: Tiempo de fin en segundos.
    :param interval: Intervalo de tiempo en segundos entre cada extracción de imagen.
    """
    
    # Verificar que el archivo de video existe
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"El archivo de video no se encontró: {video_path}")
    
    # Crear la carpeta de salida si no existe
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Abrir el video
    cap = cv2.VideoCapture(str(video_file))
    
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")
    
    # Obtener la tasa de cuadros por segundo (fps) del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("La tasa de cuadros por segundo (FPS) del video es cero.")
    
    # Convertir los tiempos de inicio y fin a números de cuadro
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    interval_frames = int(interval * fps)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validar los cuadros de inicio y fin
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    # Establecer la posición inicial del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    image_num = 1
    
    while current_frame <= end_frame:
        # Leer el cuadro
        ret, frame = cap.read()
        
        if not ret:
            print(f"No se pudo leer el cuadro en la posición {current_frame}.")
            break
        
        # Guardar la imagen
        image_name = output_dir / f"image_{image_num:05d}.jpg"
        cv2.imwrite(str(image_name), frame)
        print(f"Guardado: {image_name}")
        
        # Incrementar el contador
        current_frame += interval_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        image_num += 1
    
    # Liberar el objeto de captura
    cap.release()
    print("Extracción de imágenes completada.")

# Ejemplo de uso
if __name__ == "__main__":
    video_path = "ruta/al/video.mp4"
    output_path = "ruta/a/carpeta_de_salida"
    start_time = 10    # en segundos
    end_time = 50      # en segundos
    interval = 5       # en segundos
    
    extract_images_from_video(video_path, output_path, start_time, end_time, interval)