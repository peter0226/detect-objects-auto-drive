import cv2
import yt_dlp
import numpy as np
import os

def extraer_video_youtube(url_video):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url_video, download=False)
        formats = info_dict.get('formats', None)
        for f in formats:
            if f.get('format_id', None) == '18':
                url_video_descarga = f.get('url', None)
                return url_video_descarga
    return None

def procesar_video(url_video):
    url_descarga = extraer_video_youtube(url_video)
    if url_descarga:
        cap = cv2.VideoCapture(url_descarga)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        script_dir = os.path.dirname(os.path.abspath(__file__))
        car_cascade_path = os.path.join(script_dir, 'haarcascade_car.xml')

        car_cascade = cv2.CascadeClassifier(car_cascade_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            (rects_people, weights_people) = hog.detectMultiScale(small_frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            cars = car_cascade.detectMultiScale(gray, 1.1, 1)

            for (x, y, w, h) in rects_people:
                cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(small_frame, "Peaton", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for (x, y, w, h) in cars:
                cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(small_frame, "Vehiculo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Video", small_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No se pudo obtener la URL de descarga del video.")

url_video_youtube = "https://www.youtube.com/watch?v=ejemploDeVideoDeConduccion"
procesar_video(url_video_youtube)