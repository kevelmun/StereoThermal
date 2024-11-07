import cv2
import os
import datetime
import tkinter as tk
from PIL import Image, ImageTk

# Crear carpetas para las capturas si no existen
os.makedirs("captures/thermal", exist_ok=True)
os.makedirs("captures/visible/left", exist_ok=True)
os.makedirs("captures/visible/right", exist_ok=True)
os.makedirs("captures/video", exist_ok=True)  # Carpeta para guardar videos

class VideoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Video y Fotos")
        self.recording = False
        self.capturing = True  # Controla el bucle de captura

        # Hacer que la ventana no sea redimensionable
        self.root.resizable(False, False)

        # Crear los objetos de captura de video
        #IMPORTANT: DEPENDIENDO DE COMO SE CONECTEN LAS CAMARAS
        # SE DEBERA CAMBIAR EL ID 0,1,2...
        self.cap_visible = cv2.VideoCapture(0)
        self.cap_thermal = cv2.VideoCapture(1)

        if not self.cap_visible.isOpened():
            print("No se pudo abrir la cámara visible.")
            self.root.destroy()
            return

        if not self.cap_thermal.isOpened():
            print("No se pudo abrir la cámara térmica.")
            self.root.destroy()
            return

        # Establecer la resolución de la cámara visible a 3840x1080
        # Utilizar esto si se quiere utilziar 1920x1080
        # self.cap_visible.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        # self.cap_visible.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


        # Establecer la resolución de la cámara visible a 1280x480
        # Utilizar esto si se quiere utilziar 640x480
        self.cap_visible.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap_visible.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Variables para grabación de video
        self.out_left = None
        self.out_right = None
        self.out_thermal = None

        # Crear los widgets de la interfaz
        self.create_widgets()

        # Iniciar el hilo de actualización de video
        self.update_video()

    def create_widgets(self):
        # Frame principal
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Frame superior para botones y mensajes
        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(pady=5)

        # Frame para los botones en el lado izquierdo
        self.frame_buttons = tk.Frame(self.top_frame)
        self.frame_buttons.pack(side=tk.LEFT, padx=5)

        # Botones
        self.btn_capture = tk.Button(self.frame_buttons, text="Tomar Foto", command=self.capture_image)
        self.btn_capture.pack(pady=1)

        self.btn_record = tk.Button(self.frame_buttons, text="Iniciar Grabación", command=self.toggle_recording)
        self.btn_record.pack(pady=1)

        self.btn_exit = tk.Button(self.frame_buttons, text="Salir", command=self.exit_app)
        self.btn_exit.pack(pady=1)

        # Label para mensajes en el lado derecho
        self.message_label = tk.Label(self.top_frame, text="", fg="green", font=("Arial", 12))
        self.message_label.pack(side=tk.LEFT, padx=20)

        # Frame para las imágenes
        self.frame_images = tk.Frame(self.main_frame)
        self.frame_images.pack()

        # Marcos para las imágenes
        self.frame_left = tk.Label(self.frame_images)
        self.frame_left.grid(row=0, column=0)

        self.frame_right = tk.Label(self.frame_images)
        self.frame_right.grid(row=0, column=1)

        self.frame_thermal = tk.Label(self.frame_images)
        self.frame_thermal.grid(row=1, column=0, columnspan=2)  # Ocupa dos columnas

    def update_video(self):
        if not self.capturing:
            return

        # Leer la imagen de la cámara visible
        ret_visible, img_visible = self.cap_visible.read()
        if not ret_visible:
            print("No se pudo capturar la imagen de la cámara visible.")
            self.capturing = False
            return

        # Dividir la imagen visible en dos imágenes: izquierda y derecha
        height, width, _ = img_visible.shape
        mid_width = width // 2  # Mitad del ancho

        # Imagen izquierda
        img_left = img_visible[:, :mid_width]

        # Imagen derecha
        img_right = img_visible[:, mid_width:]

        # Leer la imagen de la cámara térmica
        ret_thermal, img_thermal = self.cap_thermal.read()
        if not ret_thermal:
            print("No se pudo capturar la imagen de la cámara térmica.")
            self.capturing = False
            return

        # Convertir la imagen térmica a escala de grises
        img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
        # Convertir de nuevo a BGR para guardar video en color
        img_thermal_bgr = cv2.cvtColor(img_thermal_gray, cv2.COLOR_GRAY2BGR)

        # Si estamos grabando, escribir los fotogramas en los archivos de video
        if self.recording:
            self.out_left.write(img_left)
            self.out_right.write(img_right)
            self.out_thermal.write(img_thermal_bgr)

        # Convertir las imágenes para Tkinter
        img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        img_thermal_rgb = cv2.cvtColor(img_thermal_gray, cv2.COLOR_GRAY2RGB)

        img_left_pil = Image.fromarray(img_left_rgb)
        img_right_pil = Image.fromarray(img_right_rgb)
        img_thermal_pil = Image.fromarray(img_thermal_rgb)

        # Redimensionar las imágenes para la interfaz, manteniendo el aspect ratio
        # Imágenes visibles (16:9)
        # Usar esto si se quiere usar la resolucion de 1920x1080
        # img_left_pil = img_left_pil.resize((320, 180), Image.LANCZOS)
        # img_right_pil = img_right_pil.resize((320, 180), Image.LANCZOS)

        #Usar esto si se quiere utilizar la resolucion de 640x480
        img_left_pil = img_left_pil.resize((320, 240), Image.LANCZOS)
        img_right_pil = img_right_pil.resize((320, 240), Image.LANCZOS)

        # Imagen térmica (4:3)
        img_thermal_pil = img_thermal_pil.resize((320, 240), Image.LANCZOS)

        self.img_left_tk = ImageTk.PhotoImage(image=img_left_pil)
        self.img_right_tk = ImageTk.PhotoImage(image=img_right_pil)
        self.img_thermal_tk = ImageTk.PhotoImage(image=img_thermal_pil)

        self.frame_left.configure(image=self.img_left_tk)
        self.frame_right.configure(image=self.img_right_tk)
        self.frame_thermal.configure(image=self.img_thermal_tk)

        # Programar la siguiente actualización
        self.root.after(10, self.update_video)

    def capture_image(self):
        # Obtener la fecha y hora actual en formato YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Definir los nombres de archivo con la marca de tiempo y los identificadores LEFT_ y RIGHT_
        left_filename = f"captures/visible/left/LEFT_visible_{timestamp}.png"
        right_filename = f"captures/visible/right/RIGHT_visible_{timestamp}.png"
        thermal_filename = f"captures/thermal/thermal_{timestamp}.png"

        # Guardar las imágenes
        ret_visible, img_visible = self.cap_visible.read()
        ret_thermal, img_thermal = self.cap_thermal.read()

        if ret_visible and ret_thermal:
            # Dividir la imagen visible en dos imágenes: izquierda y derecha
            height, width, _ = img_visible.shape
            mid_width = width // 2  # Mitad del ancho

            # Imagen izquierda
            img_left = img_visible[:, :mid_width]

            # Imagen derecha
            img_right = img_visible[:, mid_width:]

            # Convertir la imagen térmica a escala de grises
            img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(left_filename, img_left)
            cv2.imwrite(right_filename, img_right)
            cv2.imwrite(thermal_filename, img_thermal_gray)
            # Actualizar mensaje en la interfaz
            self.message_label.config(text="¡Capturas guardadas!", fg="green")
        else:
            # Actualizar mensaje de error
            self.message_label.config(text="Error al capturar imágenes.", fg="red")

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            # Obtener la fecha y hora actual en formato YYYYMMDD_HHMMSS
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Definir los nombres de archivo para los videos
            left_video_filename = f"captures/video/LEFT_video_{timestamp}.avi"
            right_video_filename = f"captures/video/RIGHT_video_{timestamp}.avi"
            thermal_video_filename = f"captures/video/THERMAL_video_{timestamp}.avi"

            # Definir el codec y crear los objetos VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes cambiar el codec si lo deseas

            # Obtener el tamaño de los fotogramas
            ret_visible, img_visible = self.cap_visible.read()
            ret_thermal, img_thermal = self.cap_thermal.read()

            if ret_visible and ret_thermal:
                # Dividir la imagen visible en dos imágenes: izquierda y derecha
                height, width, _ = img_visible.shape
                mid_width = width // 2  # Mitad del ancho

                # Imagen izquierda
                img_left = img_visible[:, :mid_width]

                # Imagen derecha
                img_right = img_visible[:, mid_width:]

                # Convertir la imagen térmica a escala de grises y luego a BGR
                img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
                img_thermal_bgr = cv2.cvtColor(img_thermal_gray, cv2.COLOR_GRAY2BGR)

                frame_size_left = (img_left.shape[1], img_left.shape[0])
                frame_size_right = (img_right.shape[1], img_right.shape[0])
                frame_size_thermal = (img_thermal_bgr.shape[1], img_thermal_bgr.shape[0])

                # Crear los VideoWriter
                self.out_left = cv2.VideoWriter(left_video_filename, fourcc, 20.0, frame_size_left)
                self.out_right = cv2.VideoWriter(right_video_filename, fourcc, 20.0, frame_size_right)
                self.out_thermal = cv2.VideoWriter(thermal_video_filename, fourcc, 20.0, frame_size_thermal)

                self.btn_record.config(text="Detener Grabación")
                # Actualizar mensaje en la interfaz
                self.message_label.config(text="Grabación iniciada...", fg="green")
            else:
                # Actualizar mensaje de error
                self.message_label.config(text="Error al iniciar grabación.", fg="red")
                self.recording = False
        else:
            # Liberar los objetos VideoWriter
            if self.out_left is not None:
                self.out_left.release()
                self.out_right.release()
                self.out_thermal.release()
                self.out_left = None
                self.out_right = None
                self.out_thermal = None
                self.btn_record.config(text="Iniciar Grabación")
                # Actualizar mensaje en la interfaz
                self.message_label.config(text="¡Videos guardados!", fg="green")

    def exit_app(self):
        # Detener la captura y cerrar la aplicación
        self.capturing = False
        self.cap_visible.release()
        self.cap_thermal.release()
        if self.recording:
            self.out_left.release()
            self.out_right.release()
            self.out_thermal.release()
        self.root.quit()

# Crear la ventana principal
root = tk.Tk()
app = VideoCaptureApp(root)
root.mainloop()
