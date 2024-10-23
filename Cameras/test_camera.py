import cv2
import os
from datetime import datetime

def open_camera(camera_index=0, desired_width=None, desired_height=None):
    """
    Abre la cámara con el índice especificado y establece la resolución deseada si se proporciona.

    :param camera_index: Índice de la cámara a abrir.
    :param desired_width: Ancho deseado de la resolución.
    :param desired_height: Alto deseado de la resolución.
    :return: Objeto de captura de video o None si falla.
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con el índice {camera_index}")
        return None
    else:
        if desired_width and desired_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width != desired_width or actual_height != desired_height:
                print(f"Resolución solicitada {desired_width}x{desired_height} no soportada.")
                print(f"Usando resolución por defecto: {actual_width}x{actual_height}")
        else:
            if camera_index == 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Cámara con índice {camera_index} seleccionada")
        print(f"Resolución actual de la cámara (ancho x alto): {int(width)} x {int(height)}")
        return cap

def display_camera_feed_and_capture(cap, is_stereo=False):
    """
    Muestra el feed de la cámara y permite capturar imágenes al presionar 'c'.

    :param cap: Objeto de captura de video.
    :param is_stereo: Booleano que indica si la cámara es estéreo.
    """
    # Crear la carpeta 'captures' si no existe
    if not os.path.exists('captures'):
        os.makedirs('captures')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir imagen de la cámara. Saliendo...")
            break

        cv2.imshow('Vista de la cámara', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Obtener la fecha y hora actual
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            if is_stereo:
                height, width, _ = frame.shape
                mid_width = width // 2
                left_image = frame[:, :mid_width]
                right_image = frame[:, mid_width:]

                # Guardar las imágenes izquierda y derecha
                left_filename = os.path.join('captures', f"LEFT_{timestamp}.png")
                right_filename = os.path.join('captures', f"RIGHT_{timestamp}.png")
                cv2.imwrite(left_filename, left_image)
                cv2.imwrite(right_filename, right_image)
                print(f"Capturas estéreo guardadas: {left_filename}, {right_filename}")
            else:
                # Guardar la imagen completa
                filename = os.path.join('captures', f"capture_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"Captura guardada: {filename}")

        # Salir con la tecla 'q'
        if key == ord('q'):
            break

def release_resources(cap):
    """
    Libera los recursos y cierra las ventanas.

    :param cap: Objeto de captura de video.
    """
    cap.release()
    cv2.destroyAllWindows()

def test_resolutions(camera_index=0):
    """
    Prueba una lista de resoluciones comunes y muestra cuáles son soportadas por la cámara.

    :param camera_index: Índice de la cámara a probar.
    """
    # Crear objeto de captura de video
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con el índice {camera_index}")
        return

    # Lista de resoluciones comunes para probar
    common_resolutions = [
        (3840, 1080),
        (160, 120),    # QVGA
        (320, 240),    # QVGA
        (640, 480),    # VGA
        (800, 600),    # SVGA
        (1024, 768),   # XGA
        (1280, 720),   # HD 720p
        (1920, 1080),  # Full HD 1080p
        (2560, 1440),  # QHD
        (3840, 2160),  # 4K UHD
    ]

    supported_resolutions = []

    print("Probando resoluciones soportadas...")
    # Probar cada resolución
    for (width, height) in common_resolutions:
        # Establecer la resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Leer el tamaño real establecido
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            supported_resolutions.append((width, height))
            print(f"Resolución soportada: {width} x {height}")
        else:
            print(f"Resolución no soportada: {width} x {height}")

    # Liberar recursos
    cap.release()

    # Mostrar las resoluciones soportadas
    print("Resoluciones soportadas por la cámara:", supported_resolutions)

def get_user_resolution():
    """
    Solicita al usuario que ingrese la resolución deseada.

    :return: Tuple de (ancho, alto) o (None, None) si el usuario no desea especificar.
    """
    while True:
        response = input("¿Desea establecer una resolución personalizada? (s/n): ").strip().lower()
        if response == 's':
            try:
                width = int(input("Ingrese el ancho deseado (píxeles): "))
                height = int(input("Ingrese el alto deseado (píxeles): "))
                return width, height
            except ValueError:
                print("Entrada inválida. Por favor, ingrese números enteros para ancho y alto.")
        elif response == 'n':
            return None, None
        else:
            print("Respuesta no válida. Por favor, ingrese 's' para sí o 'n' para no.")

def main():
    """
    Función principal que maneja el flujo del programa.
    """
    # Permitir al usuario ingresar el índice de la cámara
    try:
        camera_index = int(input("Ingrese el índice de la cámara que desea usar: "))
    except ValueError:
        print("Índice de cámara inválido. Usando 0 por defecto.")
        camera_index = 0

    # Opcional: permitir al usuario elegir la opción
    print("\nSeleccione una opción:")
    print("1. Capturar imágenes de la cámara")
    print("2. Probar resoluciones soportadas")
    choice = input("Ingrese su elección (1/2): ")

    if choice == '1':
        # Solicitar la resolución deseada
        desired_width, desired_height = get_user_resolution()

        # Preguntar si la cámara es estéreo
        stereo_input = input("¿La cámara es estéreo? (s/n): ").strip().lower()
        is_stereo = stereo_input == 's'

        # Abrir la cámara con la resolución deseada
        cap = open_camera(camera_index, desired_width, desired_height)
        if cap:
            display_camera_feed_and_capture(cap, is_stereo=is_stereo)
            release_resources(cap)
    elif choice == '2':
        test_resolutions(camera_index)
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    main()
