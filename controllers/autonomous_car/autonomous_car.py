# autonomous_car.py (リファクタリング最終版)
import os, sys, math, numpy as np
import csv
import datetime
import argparse
import atexit # atexitをインポート
from controller import Robot, Lidar, GPS, Display
from vehicle import Driver
# 分割したファイルからクラスをインポート
from utils.log_manager import LogManager
from modes.mode_line_follow import LineFollowMode
from modes.mode_cv_lane_follow import CVLaneFollowMode
from modes.mode_gemini import GeminiMode
from modes.mode_cv_lane_gemini import CVGeminiHybridMode

# ==============================================================================
# --- ⚙️ 設定エリア ---
DRIVING_MODE = 'LINE_FOLLOW' #LINE_FOLLOW,CV_LANE_FOLLOW,GEMINI
RUN_ID = 0
ENABLE_COLLISION_AVOIDANCE = False
START_Y_THRESHOLD = -26.0; START_X_MIN = 36.0; START_X_MAX = 54.0
GOAL_Y_THRESHOLD = -34.0; GOAL_X_MIN = 36.0; GOAL_X_MAX = 54.0
LAP_FINISH_MIN_TIME = 30.0; TIMEOUT_SECONDS = 120.0
#制御周期
TIME_STEP = 50
#TIME_STEP = 16
#初期（最高）速度
INITIAL_SPEED = 30.0
GEMINI_API_KEY_FILENAME = ".env"; API_CALL_INTERVAL_SEC = 2.0

gemini_mode_instance = None 

# ==============================================================================

class VehicleController:
    def __init__(self, driver: Driver):
        self.driver = driver; self.mode_name = DRIVING_MODE
        print(f"✅ 運転モード '{self.mode_name}' で起動します。")
        self.steering_angle, self.speed, self.last_speed_kmh, self.last_pos_y = 0.0, 0.0, 0.0, -9999.0
        self.is_logging_active = False; self.lap_start_time = 0.0; self.has_finished = False; self.was_in_finish_zone = False
        self._init_sensors()
        self.final_log_done = False
        self.log_manager = LogManager(mode=self.mode_name, run_id=RUN_ID)

        if self.mode_name == 'LINE_FOLLOW': self.driving_logic = LineFollowMode(INITIAL_SPEED,False)
        elif self.mode_name == 'CV_LANE_FOLLOW':
            self.driving_logic = CVLaneFollowMode(self.camera, INITIAL_SPEED, False, './images/cv_lane')
        elif self.mode_name == 'GEMINI':
            os.makedirs('./images/hybrid', exist_ok=True)
            #gemini_mode_instance = GeminiMode(self.camera, GEMINI_API_KEY_FILENAME, INITIAL_SPEED, API_CALL_INTERVAL_SEC, True, './images/gemini') 
            #driving_logic = CVGeminiHybridMode(self.camera, gemini_mode_instance, INITIAL_SPEED)
            self.driving_logic = CVGeminiHybridMode(self.camera,GEMINI_API_KEY_FILENAME,INITIAL_SPEED, API_CALL_INTERVAL_SEC,save_artifacts=False)
        else: raise ValueError("無効な運転モードです。")

        # ✅ 実験環境ログの書き込み
        experiment_log_path = os.path.join("logs", "experiment_config_log.csv")
        os.makedirs("logs", exist_ok=True)

        file_exists = os.path.isfile(experiment_log_path)
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(experiment_log_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'run_id', 'mode_name',
                    'initial_speed',         # ユーザーが指定した元の目標速度
                    'base_initial_speed',    # 実際に使われたランダム化後の速度
                    'initial_steering',
                    'log_file'
                ])
            writer.writerow([
                now_str,
                RUN_ID,
                self.mode_name,
                getattr(self.driving_logic, 'initial_speed', None),
                getattr(self.driving_logic, 'base_initial_speed', None),
                getattr(self.driving_logic, 'initial_steering', None),
                getattr(self.log_manager, 'log_file_path', '')
            ])

        # ✅ 初期速度は「ランダム後」の速度で set
        self.set_speed(getattr(self.driving_logic, 'base_initial_speed', INITIAL_SPEED))
        #self.set_speed(INITIAL_SPEED)


    def _init_sensors(self):
        self.camera = self.driver.getDevice("camera"); self.camera.enable(TIME_STEP)
        self.gps = self.driver.getDevice("gps"); self.gps.enable(TIME_STEP)
        self.display = self.driver.getDevice("display")
        if self.display:
            try: self.speedometer_image = self.display.imageLoad("speedometer.png")
            except: self.display = None

        global ENABLE_COLLISION_AVOIDANCE # global宣言を関数の先頭に移動
        if ENABLE_COLLISION_AVOIDANCE:
            self.lidar = self.driver.getDevice("Sick LMS 291")
            if self.lidar: self.lidar.enable(TIME_STEP); print("✅ 障害物回避Lidarが有効です。")
            else: print("警告: Lidarが見つかりません。"); ENABLE_COLLISION_AVOIDANCE = False

    def run_step(self):
        if self.has_finished: 
            self.driver.setBrakeIntensity(1.0); 
            self.set_speed(0); 
            if not self.final_log_done:
                self._log_and_display()
                self.final_log_done = True
            
            return False

        self._update_lap_status()
        if self.mode_name == 'GEMINI': 
            #image_bytes = self.camera.getImage()
            #with self.driving_logic.lock:
            #    self.driving_logic.shared_image_bytes = image_bytes

            current_speed_kmh = self.driver.getCurrentSpeed() 
            proposed_steer, proposed_speed, brake = self.driving_logic.get_command(self.camera, current_speed_kmh)

            #proposed_steer, proposed_speed, brake = self.driving_logic.get_command(self.camera)

        else: 
            proposed_steer, proposed_speed, brake = self.driving_logic.get_command(self.camera)
        
        final_steer, final_speed = proposed_steer, proposed_speed
        if brake: self.driver.setBrakeIntensity(0.8)
        else: self.driver.setBrakeIntensity(0.0)
        self.set_speed(final_speed); self.set_steering_angle(final_steer)
        self._log_and_display()

        return True

    def _update_lap_status(self):
        current_time = self.driver.getTime(); pos_x, pos_y = self.gps.getValues()[:2]
        if not self.is_logging_active:
            if (self.last_pos_y <= START_Y_THRESHOLD and pos_y > START_Y_THRESHOLD and START_X_MIN < pos_x < START_X_MAX):
                self.is_logging_active, self.lap_start_time = True, current_time; self.log_manager.start_logging(); print(f"🏁 スタート！")
        else:
            lap_time = current_time - self.lap_start_time
            if lap_time > TIMEOUT_SECONDS: print(f"⏰ タイムアウト"); self.has_finished = True
            if lap_time > LAP_FINISH_MIN_TIME and (self.last_pos_y <= GOAL_Y_THRESHOLD and pos_y > GOAL_Y_THRESHOLD and GOAL_X_MIN < pos_x < GOAL_X_MAX):
                print(f"🎉 ゴール！ラップタイム: {lap_time:.2f} 秒"); self.has_finished = True
        self.last_pos_y = pos_y

    def _log_and_display(self):
        if self.is_logging_active:
            # === 各種値の取得 ===
            current_time = self.driver.getTime()
            current_speed_kmh = self.driver.getCurrentSpeed() # km/h
            #acceleration = (current_speed_ms - self.last_speed_ms) / (TIME_STEP / 1000.0) if self.last_speed_ms > 0 else 0
            current_speed_ms = current_speed_kmh / 3.6          # m/s に変換
            last_speed_ms = self.last_speed_kmh / 3.6  # 単位変換：km/h → m/s


            acceleration = (
                (current_speed_ms - last_speed_ms) / (TIME_STEP / 1000.0)
                if last_speed_ms > 0 else 0
            )

            gps_x, gps_y = self.gps.getValues()[:2]
            actual_steering = self.driver.getSteeringAngle()
            error_angle = actual_steering - self.steering_angle

            # === ログ記録 ===
            log_data = {
                "timestamp": current_time,
                "lap_time": current_time - self.lap_start_time,
                "pos_x": gps_x,
                "pos_y": gps_y,
                "speed_kmh": current_speed_kmh,
                "target_speed_kmh": self.speed,
                "steering_angle": actual_steering,
                "target_steering_angle": self.steering_angle,
                "acceleration": acceleration,
                "mode_name": self.mode_name,
                "run_id": RUN_ID,
                "is_goal": int(self.has_finished),
                "is_logging_active": int(self.is_logging_active),
                "error_angle": error_angle,
                # "control_latency": self.latest_latency  # ← run_step内で記録が必要（今後対応）
            }
            self.log_manager.log_step(log_data)

        self.last_speed_kmh = self.driver.getCurrentSpeed()

        # === 表示処理 ===
        if self.display and self.speedometer_image:
            self.display.imagePaste(self.speedometer_image, 0, 0, False)
            speed = self.driver.getCurrentSpeed()
            speed = 0 if math.isnan(speed) else speed
            alpha = speed / 260.0 * 3.72 - 0.27
            x, y = -int(50.0 * math.cos(alpha)), -int(50.0 * math.sin(alpha))
            self.display.drawLine(100, 95, 100 + x, 95 + y)
            coords = self.gps.getValues()
            #gps_speed = self.gps.getSpeed() * 3.6
            self.display.drawText(f"GPS: {coords[0]:.1f} {coords[1]:.1f}", 10, 130)
            self.display.drawText(f"Speed: {speed:.1f} km/h", 10, 140)
            if self.is_logging_active and not self.has_finished:
                self.display.drawText(f"Lap Time: {self.driver.getTime() - self.lap_start_time:.1f}s", 10, 120)


    def set_speed(self, kmh): self.speed = np.clip(kmh, 0, 100); self.driver.setCruisingSpeed(self.speed)
    def set_steering_angle(self, wheel_angle): self.steering_angle = np.clip(wheel_angle, -0.6, 0.6); self.driver.setSteeringAngle(self.steering_angle)
    
    # クリーンアップ関数を定義
    def perform_cleanup():
        global gemini_mode_instance
        if gemini_mode_instance:
            gemini_mode_instance.cleanup() # GeminiModeのcleanupメソッドを呼び出す


    # プログラム終了時にperform_cleanupを呼び出すように登録
    atexit.register(perform_cleanup)

    def close(self):
         self.log_manager.close()

if __name__ == "__main__":

    # 環境変数からモードと試行番号を取得（なければデフォルト値）
    strategy_env = os.environ.get("STRATEGY_NAME", "GEMINI")
    trial_env = os.environ.get("TRIAL_NUMBER", "0")

    # argparseと併用可能（環境変数優先）
    parser = argparse.ArgumentParser(description="自動運転モード実行")
    parser.add_argument('--mode', type=str, default=strategy_env,
                        choices=['LINE_FOLLOW', 'CV_LANE_FOLLOW', 'GEMINI'],
                        help="運転モード")
    parser.add_argument('--run_id', type=int, default=int(trial_env),
                        help="試行番号")
    args = parser.parse_args()

    # ここで引数をグローバルに反映
    DRIVING_MODE = args.mode
    RUN_ID = args.run_id

    robot_driver = Driver()
    controller = None
    try:
        controller = VehicleController(driver=robot_driver)
        while robot_driver.step() != -1:
            #controller.run_step()
            if not controller.run_step():
                break
    except Exception as e:
        print(f"致命的なエラーが発生しました: {e}", file=sys.stderr)
    finally:
        
        if controller: controller.close()