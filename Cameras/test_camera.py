import cv2

def open_camera(camera_index=0):
    # Crear el objeto de captura de video
    cap = cv2.VideoCapture(camera_index)
    # Comprobar si la cámara se abrió correctamente
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con el índice {camera_index}")
        return None
    else:
        if camera_index == 0:
          
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Cámara con índice {camera_index} seleccionada")
        print(f"Resolución actual de la cámara (ancho x alto): {int(width)} x {int(height)}")
        return cap

def display_camera_feed(cap):
    # Mostrar la imagen de la cámara
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir imagen de la cámara. Saliendo...")
            break

        cv2.imshow('Vista de la cámara', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def release_resources(cap):
    # Liberar los recursos y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

def test_resolutions(camera_index=0):
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

def main():
    # Permitir al usuario ingresar el índice de la cámara
    camera_index = int(input("Ingrese el índice de la cámara que desea usar: "))

    # Opcional: permitir al usuario elegir la opción
    print("Seleccione una opción:")
    print("1. Mostrar feed de la cámara")
    print("2. Probar resoluciones soportadas")
    choice = input("Ingrese su elección (1/2): ")

    if choice == '1':
        cap = open_camera(camera_index)
        if cap:
            display_camera_feed(cap)
            release_resources(cap)
    elif choice == '2':
        test_resolutions(camera_index)
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    main()
