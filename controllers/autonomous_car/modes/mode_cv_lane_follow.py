# modes/mode_cv_lane_follow.py (交差点対応版)
import numpy as np
import cv2
import os
import datetime
from .base_mode import BaseMode

class CVLaneFollowMode(BaseMode):
    def __init__(self, camera, initial_speed, save_images=False, save_dir='./images/cv_lane'):

        super().__init__(initial_speed)

        self.initial_speed = initial_speed
        self.camera_height = camera.getHeight()
        self.camera_width = camera.getWidth()
        self.M, self.invM = self._calculate_perspective_transform()
        self.save_images = save_images
        self.save_dir = save_dir
        
        # 状態を記憶するための変数
        self.last_left_base = None
        self.last_right_base = None
        self.LANE_WIDTH_PIXELS = 350
        
        # ✅ --- 線を見失った時間をカウントする変数を追加 ---
        self.lost_line_counter = 0
        
        print("✅ CVレーン検出モードの準備完了。画像保存:", "有効" if save_images else "無効")

    def _calculate_perspective_transform(self):
        h, w = self.camera_height, self.camera_width
        src = np.float32([[w * 0.15, h * 0.7], [w * 0.85, h * 0.7], [w, h], [0, h]])
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)
    
    def get_command(self, camera):

        initial = self.get_initial_command()
        if initial:
            return initial

        image_bytes = camera.getImage()
        if not image_bytes: return 0.0, 0.0, True

        h, w = self.camera_height, self.camera_width
        img = np.frombuffer(image_bytes, np.uint8).reshape((h, w, 4))
        bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges_img = cv2.Canny(blur_img, 50, 150)
        warped_img = cv2.warpPerspective(edges_img, self.M, (w, h), flags=cv2.INTER_LINEAR)
        
        histogram = np.sum(warped_img[h//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        
        left_detected = histogram[left_base] > 300
        right_detected = histogram[right_base] > 300
        
        lane_center = 0

        # --- ✅ 判断ロジック ---
        if left_detected and right_detected:
            # ケース1: 両方検出できた場合 (正常)
            self.lost_line_counter = 0 # カウンターをリセット
            self.last_left_base, self.last_right_base = left_base, right_base
            lane_center = (left_base + right_base) / 2
        elif left_detected and self.last_right_base is not None:
            # ケース2: 左側のみ検出
            self.lost_line_counter = 0
            self.last_left_base = left_base
            lane_center = (left_base + (left_base + self.LANE_WIDTH_PIXELS)) / 2
        elif right_detected and self.last_left_base is not None:
            # ケース3: 右側のみ検出
            self.lost_line_counter = 0
            self.last_right_base = right_base
            lane_center = ((right_base - self.LANE_WIDTH_PIXELS) + right_base) / 2
        else:
            # ケース4: 両方見えない場合 (交差点と判断)
            self.lost_line_counter += 1
            if self.lost_line_counter < 500: # 約25秒間(500ステップ)は直進を試みる
                # ハンドルをまっすぐにし、少し減速して直進
                return 0.0, self.initial_speed * 0.6, False
            else:
                # 2.5秒経っても線が見つからなければ、安全のために停止
                return 0.0, 0.0, True

        offset = lane_center - midpoint
        steering_angle = offset * 0.006

        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            
            original_path = os.path.join(self.save_dir, f"{timestamp}_original.png")
            warped_path = os.path.join(self.save_dir, f"{timestamp}_warped.png")
            
            cv2.imwrite(original_path, bgr_img)
            cv2.imwrite(warped_path, warped_img)

        return steering_angle, self.initial_speed, False