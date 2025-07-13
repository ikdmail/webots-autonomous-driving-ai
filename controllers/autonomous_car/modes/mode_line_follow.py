# modes/mode_line_follow.py (復帰ロジック追加版)
import numpy as np
from numba import njit
import os
import cv2
import datetime
from .base_mode import BaseMode


# --- モード固有の定数 ---
UNKNOWN = 99999.99
PID_KP, PID_KI, PID_KD = 0.25, 0.006, 2
FILTER_SIZE = 3

# (Numbaで高速化されたヘルパー関数は変更なし)
@njit(fastmath=True)
def _color_diff(pixel_rgb, ref_rgb):
    return abs(pixel_rgb[0]-ref_rgb[0]) + abs(pixel_rgb[1]-ref_rgb[1]) + abs(pixel_rgb[2]-ref_rgb[2])
@njit(fastmath=True)
def _process_image(image_array, width, height, fov):
    REF_RGB = (95, 187, 203)
    sum_x, pixel_count = 0, 0
    start_y = int(height * 0.6)  # 下40%に限定


    for y in range(start_y, height):
        for x in range(width):
            pixel_color = (image_array[y, x, 0], image_array[y, x, 1], image_array[y, x, 2])
            if _color_diff(pixel_color, REF_RGB) < 30:
                sum_x += x
                pixel_count += 1
    if pixel_count == 0:
        return UNKNOWN
    return (float(sum_x) / pixel_count / width - 0.5) * fov

class LineFollowMode(BaseMode):
    #def __init__(self, initial_speed):
    def __init__(self, initial_speed, save_images=False, save_dir='./images/line_follow'):
        super().__init__(initial_speed)

        self.initial_speed = initial_speed
        self.filter_old_value = [0.0] * FILTER_SIZE
        self.filter_first_call = True
        self.pid_need_reset = True
        self.pid_old_value = 0.0
        self.pid_integral = 0.0
        self.lost_count = 0  # 線を見失ったフレームの連続回数

        self.save_images = save_images
        self.save_dir = save_dir
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)


        # ✅ --- 最後に有効だった操舵角を記憶する変数を追加 ---
        self.last_known_steering = 0.0
        print("✅ 黄線追従モードの準備完了。")

    def _filter_angle(self, new_value):
        if self.filter_first_call or new_value == UNKNOWN:
            self.filter_first_call = False; self.filter_old_value = [0.0] * FILTER_SIZE
        else:
            self.filter_old_value.pop(0)
            self.filter_old_value.append(new_value if new_value != UNKNOWN else 0.0)
        if new_value == UNKNOWN: return UNKNOWN
        filtered_values = [v for v in self.filter_old_value if v != 0.0]
        if not filtered_values: return UNKNOWN
        return sum(filtered_values) / len(filtered_values)

    def _apply_pid(self, angle):
        if self.pid_need_reset:
            self.pid_old_value = angle; self.pid_integral = 0.0; self.pid_need_reset = False
        diff = angle - self.pid_old_value
        self.pid_integral = np.clip(self.pid_integral + angle, -30, 30)
        self.pid_old_value = angle
        return (PID_KP * angle) + (PID_KI * self.pid_integral) + (PID_KD * diff)

    def get_command(self, camera):

        initial = self.get_initial_command()
        if initial:
            return initial

        image_bytes = camera.getImage()
        if not image_bytes:
            return 0.0, 0.0, True

        w, h, fov = camera.getWidth(), camera.getHeight(), camera.getFov()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((h, w, 4))
        
        raw_angle = _process_image(image_array, w, h, fov)
        yellow_line_angle = self._filter_angle(raw_angle)

        if self.save_images:
            # BGRA → BGR に変換
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)

            # 下40% を切り出し
            start_y = int(h * 0.6)
            cropped = bgr_image[start_y:h, :]

            # 保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            filename = os.path.join(self.save_dir, f'{timestamp}_bottom40.png')
            cv2.imwrite(filename, cropped)
        
        if yellow_line_angle != UNKNOWN:
            # --- 線を検出できた場合 ---
            steering_angle = self._apply_pid(yellow_line_angle)
            self.last_known_steering = steering_angle # 正常な操舵角を記憶
            self.lost_count = 0  # ✅ リセット

            return steering_angle, self.initial_speed, False
        else:
            # --- ✅ 線を見失った場合の復帰ロジック ---
            self.pid_need_reset = True

            self.lost_count += 1  # ✅ インクリメント

            if self.lost_count < 4:
                return self.last_known_steering, self.initial_speed * 0.3, False
            elif self.lost_count < 10:
            # 緩やかに左右に振る探索動作
                sweep_angle = np.sin(self.lost_count * 0.5) * 0.3  # ±0.3 rad程度で探索
                return sweep_angle, self.initial_speed * 0.2, False
            else:
                return self.last_known_steering, self.initial_speed * 0.3, False