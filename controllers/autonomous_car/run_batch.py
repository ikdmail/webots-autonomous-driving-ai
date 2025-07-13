import os
import subprocess
import time

# --- 設定（★★ご自身の環境に合わせて必ず変更してください★★） ---
# Webotsの実行ファイルのパス
WEBOTS_PATH = r"C:\Users\User\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe"

# 実行したいワールドファイルのパス
# このワールドはコントローラーとして 'autonomous_car.py' を使用するよう設定されている必要があります。
WORLD_PATH = r"C:\Users\User\dev\city\worlds\city.wbt"

# --- ★★★ 実験パラメータ ★★★ ---
# 実行したい運転モード（DRIVING_MODE）のリスト
# autonomous_car.py で定義されているモード名を指定します。
MODES_TO_RUN = [
    "LINE_FOLLOW",
    "CV_LANE_FOLLOW",
    "GEMINI"  # このモードの時だけリアルタイムで実行されます
]
TOTAL_TRIALS = 10                  # 1モードあたりの総試行回数
WAIT_INTERVAL_SECONDS = 10         # 試行間の待機時間（秒）
SIMULATION_RUN_TIME_SECONDS = 140  # 1回のシミュレーション最大実行時間

def run_single_trial(mode, trial_number):
    """単一の試行を実行する"""
    print(f"\n---【モード: {mode} | 試行: {trial_number}/{TOTAL_TRIALS} を開始】---")

    # 環境変数を設定して、autonomous_car.pyにモード名と試行回数を渡す
    env = os.environ.copy()
    env['STRATEGY_NAME'] = mode
    env['TRIAL_NUMBER'] = str(trial_number)

    # ★★ モードによって実行速度を切り替える ★★
    if mode == "GEMINI":
        print("🚀 GEMINIモードを検出。--mode=realtime で実行します。")
        simulation_mode = "--mode=realtime"
    else:
        print("⚡️ 高速モード (--mode=fast) で実行します。")
        simulation_mode = "--mode=fast"

    command = [WEBOTS_PATH, "--batch", simulation_mode, WORLD_PATH]

    process = None
    try:
        print(f"Webotsを起動します...（最大実行時間: {SIMULATION_RUN_TIME_SECONDS}秒）")
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout, stderr = process.communicate(timeout=SIMULATION_RUN_TIME_SECONDS)

        print("--- Webots stdout ---")
        print(stdout)
        print("--- Webots stderr ---")
        if stderr:
            print(stderr)
        print("---------------------")

    except subprocess.TimeoutExpired:
        print("シミュレーションがタイムアウトしました。プロセスを強制終了します。")
        if process:
            process.kill()
            stdout, stderr = process.communicate()
            print("--- Webots stdout (on timeout) ---")
            print(stdout)
            print("--- Webots stderr (on timeout) ---")
            if stderr:
                print(stderr)
            print("----------------------------------")

    except FileNotFoundError:
        print(f"エラー: Webotsの実行ファイルが見つかりません。パスを確認してください: {WEBOTS_PATH}")
        return False
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        if process:
            process.kill()
        return False
    finally:
        print(f"---【モード: {mode} | 試行: {trial_number} が完了】---")

    return True

def main():
    """メイン処理"""
    total_executions = len(MODES_TO_RUN) * TOTAL_TRIALS
    current_execution = 0

    for mode in MODES_TO_RUN:
        for i in range(1, TOTAL_TRIALS + 1):
            current_execution += 1
            print(f"\n========================================================")
            print(f"実験全体進捗: {current_execution} / {total_executions}")
            print(f"========================================================")

            if not run_single_trial(mode, i):
                print("エラーのため実験を中断します。")
                return

            if current_execution < total_executions:
                print(f"\n次の試行まで {WAIT_INTERVAL_SECONDS}秒 待機します...")
                time.sleep(WAIT_INTERVAL_SECONDS)

    print("\n\n★★★★★ 全ての実験が完了しました ★★★★★")

if __name__ == '__main__':
    main()