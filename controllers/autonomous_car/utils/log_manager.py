# utils/log_manager.py
import datetime
import os

class LogManager:
    def __init__(self, mode: str, run_id: int = 0, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"log_{mode}_run{run_id}_{timestamp}.csv")
        self.log_file = None

    def start_logging2(self):
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.header = ["timestamp", "lap_time", "pos_x", "pos_y", "speed_kmh", 
                       "target_speed_kmh", "steering_angle", "target_steering_angle", 
                       "acceleration"]
        self.log_file.write(",".join(self.header) + "\n")
        print(f"📄 ログファイルを '{self.log_file_path}' に作成し、記録を開始します。")


    def start_logging(self):
        self.log_file = open(self.log_file_path, 'w', newline='', encoding='utf-8')
        self.header = [
            "timestamp", "lap_time", "pos_x", "pos_y", "speed_kmh", 
            "target_speed_kmh", "steering_angle", "target_steering_angle", 
            "acceleration", "mode_name", "run_id", "is_goal", 
            "is_logging_active", "error_angle"
        ]
        self.log_file.write(",".join(self.header) + "\n")
        print(f"📄 ログファイルを '{self.log_file_path}' に作成し、記録を開始します。")


    def log_step(self, data: dict):
        if not self.log_file:
            return
        row = [f"{data.get(h, ''):.4f}" if isinstance(data.get(h), float) else str(data.get(h, "")) for h in self.header]
        self.log_file.write(",".join(row) + "\n")

    def close(self):
        if self.log_file:
            self.log_file.close()
            print(f"🛑 ログファイル '{self.log_file_path}' を閉じました。")
