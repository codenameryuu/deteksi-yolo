import os
import cv2
import filetype
import datetime
from flask import Flask, request, render_template, Response

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()

dir_path = os.path.dirname(os.path.realpath(__file__))

# Konfogurasi lokasi upload file
app.config["FILE_UPLOADS"] = dir_path + "/uploaded_files/"


# Url utama
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload-file', methods=['GET', 'POST'])
# Url upload file
def upload_file():
    # Cek jika request method nya POST
    if request.method == "POST":
        # Cek jika ada file yang diupload
        if request.files:
            # Simpan file ke lokasi yang ditentukan
            file = request.files["file"]
            date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            filename = str(date) + '_' + file.filename
            file.save(os.path.join(
                app.config["FILE_UPLOADS"], filename))

            # Pindah halaman ke hasil.html dengan membawa parameter nama file yg diupload
            return render_template("hasil.html", filename=filename)

    # Pindah halaman ke hasil.html
    return render_template("hasil.html")


# Fungsi streaming file yang telah diupload
def stream(filename):
    # Ambil lokasi file
    file = app.config["FILE_UPLOADS"] + filename

    # Cek jika file sebuah gambar
    if filetype.is_image(file):
        # Baca file tersebut lalu langsung stream kan ke hasil.html
        img = cv2.imread(file)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # Cek jika file sebuah video
    elif filetype.is_video(file):
        # Baca file video tersebut
        cap = cv2.VideoCapture(file)

        # Lakukan perulangan sampai file video ditutup
        while(cap.isOpened()):
            # Inisiasi file
            ret, frame = cap.read()

            # Jika file habis diputar, maka ulangi
            if not ret:
                frame = file
                continue

            # Jika ada frame, maka eksekusi code berikut
            if ret:
                # Buat ulang ukuran file menjadi gambar
                image = cv2.resize(frame, (0, 0), None, 1, 1)

                # Konversi image ke mode gray (abu-abu)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Gunakan substraksi background
                fgmask = sub.apply(gray)

                # Terapkan kernel ke morfologi
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
                dilation = cv2.dilate(opening, kernel)

                # Hilangkan bayangan
                retvalbin, bins = cv2.threshold(
                    dilation, 220, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(
                    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                minarea = 400
                maxarea = 50000

                # Perulangan pengecekan contours
                for i in range(len(contours)):

                    # Gunakan hirarki hanya untuk menghitung countur induk
                    if hierarchy[0, i, 3] == -1:
                        # Area dari contour
                        area = cv2.contourArea(contours[i])

                        # Cek perbatasan area contour
                        if minarea < area < maxarea:

                            # Menghitung centroid dari contour
                            cnt = contours[i]
                            M = cv2.moments(cnt)
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])

                            # Mendapat titik batas contour untuk membuat persegi panjang
                            # x,y adalah sudut kiri atas dan w,h adalah lebar dan tinggi
                            x, y, w, h = cv2.boundingRect(cnt)

                            # Membuat persegi panjang di sekitar contour
                            cv2.rectangle(
                                image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Mencetak teks centroid untuk diperiksa ulang
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10,
                                        cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
                            cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS,
                                           markerSize=8, thickness=3, line_type=cv2.LINE_8)

            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # time.sleep(0.1)
            key = cv2.waitKey(20)
            if key == 27:
                break


@app.route('/start_steaming/<filename>')
# Url mulai streaming
def start_steaming(filename):
    return Response(stream(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
