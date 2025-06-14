import cv2
import mediapipe as mp
from ultralytics import YOLO
from djitellopy import Tello
import asyncio
import websockets
import base64

# --- Kumpulan Klien WebSocket ---
# Untuk menyiarkan frame ke semua klien yang terhubung
connected_clients = set()


# Fungsi untuk mendaftarkan klien baru
async def register(websocket):
    connected_clients.add(websocket)
    print(f"Klien baru terhubung. Total: {len(connected_clients)}")


# Fungsi untuk membatalkan pendaftaran klien
async def unregister(websocket):
    connected_clients.remove(websocket)
    print(f"Klien terputus. Total: {len(connected_clients)}")


# Fungsi untuk menyiarkan frame ke semua klien
async def broadcast_frame(frame_data):
    if connected_clients:
        # Buat tugas untuk mengirim ke semua klien secara bersamaan
        await asyncio.wait([client.send(frame_data) for client in connected_clients])


# Fungsi handler utama untuk koneksi WebSocket
async def ws_handler(websocket, path):
    await register(websocket)
    try:
        await websocket.wait_closed()
    finally:
        await unregister(websocket)


# --- Fungsi Inti Machine Learning Anda (Tidak Diubah) ---
def get_center(landmarks, indices, image_shape):
    h, w = image_shape[:2]
    points = []
    for i in indices:
        if landmarks[i].visibility > 0.5:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            points.append((x, y))
    if not points:
        return None
    avg_x = sum([p[0] for p in points]) // len(points)
    avg_y = sum([p[1] for p in points]) // len(points)
    return (avg_x, avg_y)


# --- Fungsi Utama Pemrosesan Video ---
async def video_processing_loop():
    print("Mulai inisialisasi drone dan model ML...")

    # Inisialisasi drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    # Muat model
    yolo_model = YOLO('yolov8n.pt')
    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)
    hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=2)

    # Inisialisasi modul Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    print("Inisialisasi selesai. Memulai loop utama...")

    try:
        frame_count = 0
        while True:
            frame = tello.get_frame_read().frame
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))
            output_frame = frame.copy()

            # >>> SEMUA LOGIKA PEMROSESAN YOLO & MEDIAPIPE ANDA DITARUH DI SINI <<<
            # (Kode dari 'results = yolo_model(frame, verbose=False)' sampai 'cv2.putText(output_frame, ...)')
            # ... (Saya singkat untuk kejelasan, cukup salin-tempel kode Anda di sini)
            results = yolo_model(frame, verbose=False)
            human_detected = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                            human_detected = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if human_detected:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        output_frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
            # ... (dan seterusnya)

            # Tambahkan info baterai
            battery_level = tello.get_battery()
            cv2.putText(output_frame, f"Battery: {battery_level}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Perubahan Kunci: Encode dan Siarkan Frame ---
            # 1. Encode frame yang telah diproses menjadi JPEG
            _, buffer = cv2.imencode('.jpg', output_frame)
            # 2. Ubah buffer JPEG menjadi string Base64
            frame_data = base64.b64encode(buffer).decode('utf-8')
            # 3. Siarkan ke semua klien yang terhubung
            await broadcast_frame(frame_data)

            # Beri sedikit waktu agar event loop bisa berjalan
            await asyncio.sleep(0.01)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Loop pemrosesan dihentikan.")
    finally:
        print("Membersihkan sumber daya...")
        tello.streamoff()
        tello.end()
        pose.close()
        hands.close()
        print("Selesai!")


# --- Fungsi Utama untuk Menjalankan Semuanya ---
async def main():
    # Jalankan server WebSocket di latar belakang
    websocket_server = await websockets.serve(ws_handler, "localhost", 8765)
    print("Server WebSocket berjalan di ws://localhost:8765")

    # Jalankan loop pemrosesan video
    processing_task = asyncio.create_task(video_processing_loop())

    # Tunggu kedua tugas selesai
    await asyncio.gather(processing_task, websocket_server.wait_closed())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program dihentikan oleh pengguna.")