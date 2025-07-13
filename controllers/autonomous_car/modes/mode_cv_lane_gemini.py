# modes/mode_cv_gemini_hybrid.py
import numpy as np
import cv2
import os
import datetime
import threading
from PIL import Image
import json
import time
import google.generativeai as genai
from .base_mode import BaseMode

class CVGeminiHybridMode(BaseMode):
    def __init__(self, camera, api_key_filename, initial_speed, api_call_interval, save_artifacts=False, save_dir='./images/hybrid'):
        super().__init__(initial_speed)

        self.camera = camera
        self.initial_speed = initial_speed
        self.api_call_interval = api_call_interval
        self.save_artifacts = save_artifacts
        self.save_dir = save_dir
        self.camera_height = camera.getHeight()
        self.camera_width = camera.getWidth()
        self.M, self.invM = self._calculate_perspective_transform()
        self.LANE_WIDTH_PIXELS = 350
        self.last_left_base = None
        self.last_right_base = None
        self.lost_line_counter = 0
        
        # Gemini関連
        self.shared_data = {"steering": 0.0, "speed": self.initial_speed, "new_command_ready": False}
        self.shared_image_bytes = None
        self.lock = threading.Lock()
        self.stop_worker_flag = False
        self.current_speed_kmh = 0.0

        os.makedirs(self.save_dir, exist_ok=True)
        self._init_gemini(api_key_filename)
        print("✅ ハイブリッドモード（CV+Gemini）準備完了")

    def _calculate_perspective_transform(self):
        h, w = self.camera_height, self.camera_width
        src = np.float32([[w * 0.15, h * 0.7], [w * 0.85, h * 0.7], [w, h], [0, h]])
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    def _init_gemini(self, api_key_filename):
        try:
            key_file_path = os.path.join(os.path.dirname(__file__), "..", api_key_filename)
            with open(key_file_path, 'r') as f:
                api_key = f.read().strip()
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            threading.Thread(target=self._api_worker, daemon=True).start()
        except Exception as e:
            raise RuntimeError(f"Gemini初期化に失敗: {e}")

    def _api_worker(self):
        DRIVING_PROMPT = """
        あなたは自動運転AIです。以下の画像は前方カメラの映像です。
        もし横断歩道付近に動物や人がいて、今から横断する可能性がある場合は、
        必ず速度を下げるか停止してください。
        現在の速度{speed}
        ステアリング角度 (-0.5〜0.5)、速度 (0〜{max_speed}) をJSONで返してください。
        {{
          "steering_angle": [float],
          "speed_kmh": [float]
        }}
        """.format(speed=self.current_speed_kmh,max_speed=self.initial_speed)

        while not self.stop_worker_flag:
            with self.lock:
                image_bytes = self.shared_image_bytes
            if not image_bytes or len(image_bytes) != self.camera_width * self.camera_height * 4:
                time.sleep(self.api_call_interval)
                continue

            pil_image_rgb = Image.frombytes('RGBA', (self.camera_width, self.camera_height), image_bytes).convert('RGB')
            if self.save_artifacts:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                cv2.imwrite(os.path.join(self.save_dir, f'{timestamp}_input.png'),
                            np.array(pil_image_rgb)[:, :, ::-1])

            response = self.gemini_model.generate_content([pil_image_rgb, DRIVING_PROMPT])
            print(f"🚨 Geminiレスポンス解析: {response}")

            try:
                command = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
                with self.lock:
                    self.shared_data["steering"] = float(command.get("steering_angle", 0.0))
                    self.shared_data["speed"] = float(command.get("speed_kmh", 0.0))
                    self.shared_data["new_command_ready"] = True
            except Exception as e:
                print(f"🚨 Geminiレスポンス解析失敗: {e}")

            time.sleep(self.api_call_interval)

    def get_command(self, camera,current_speed_kmh):

        initial = self.get_initial_command()
        if initial:
            return initial


        image_bytes = camera.getImage()
        if not image_bytes:
            return 0.0, 0.0, True

        # Gemini用に画像保存
        with self.lock:
            self.shared_image_bytes = image_bytes
            self.current_speed_kmh = current_speed_kmh

        h, w = self.camera_height, self.camera_width
        img = np.frombuffer(image_bytes, np.uint8).reshape((h, w, 4))
        bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        warped = cv2.warpPerspective(edges, self.M, (w, h))

        histogram = np.sum(warped[h//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        left_detected = histogram[left_base] > 300
        right_detected = histogram[right_base] > 300

        if left_detected and right_detected:
            self.lost_line_counter = 0
            self.last_left_base = left_base
            self.last_right_base = right_base
            lane_center = (left_base + right_base) / 2
        elif left_detected and self.last_right_base:
            self.lost_line_counter = 0
            self.last_left_base = left_base
            lane_center = (left_base + left_base + self.LANE_WIDTH_PIXELS) / 2
        elif right_detected and self.last_left_base:
            self.lost_line_counter = 0
            self.last_right_base = right_base
            lane_center = (right_base + right_base - self.LANE_WIDTH_PIXELS) / 2
        else:
            # 両方検出できなければGeminiに任せる
            self.lost_line_counter += 1
            if self.lost_line_counter < 50:
                return 0.0, self.initial_speed * 0.6, False
            else:
                #print(f"🚨 Gemini利用開始")
                with self.lock:
                    if self.shared_data["new_command_ready"]:
                        steer = self.shared_data["steering"]
                        speed = self.shared_data["speed"]
                        self.shared_data["new_command_ready"] = False
                        print(f"🚨 Gemini利用開始:steer={steer},speed={speed}")
                        return steer, speed, speed <= 0
                return 0.0, self.initial_speed * 0.6, True

        offset = lane_center - midpoint
        steering_angle = offset * 0.006
        return steering_angle, self.initial_speed, False

    def cleanup(self):
        print("🔴 Geminiワーカースレッド停止中...")
        self.stop_worker_flag = True
