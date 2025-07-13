import random

class BaseMode:
    def __init__(self, base_speed_kmh):
        self.base_initial_speed = self._randomize_speed(base_speed_kmh)
        self.initial_steering = self._randomize_steering()
        self.starting = True
        print(f"✅ BaseMode初期化: 初期速度={self.base_initial_speed:.2f} km/h, 初期ステアリング={self.initial_steering:.3f}")


    def _randomize_speed(self, base_speed, variation=2.0):
        return base_speed + random.uniform(-variation, variation)

    def _randomize_steering(self, variation=0.03):
        return random.uniform(-variation, variation)

    def get_initial_command(self):
        if self.starting:
            self.starting = False
            return self.initial_steering, self.base_initial_speed, False
        return None  # 初期ステップ以外
