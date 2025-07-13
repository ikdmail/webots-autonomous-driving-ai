# AI-Enhanced Autonomous Driving in Webots

This is a project for building and evaluating autonomous driving controllers within the [Webots](https://cyberbotics.com/) robotics simulator. It compares multiple control architectures, from simple algorithms to advanced AI models, and analyzes their performance limits under various conditions.

## Key Features

- **Multiple Control Models**: Implements and compares three distinct driving models:
    1.  **Line Follow**: A simple vision-based model that follows a specific colored line.
    2.  **CV Lane Follow**: A standard computer vision model that detects and follows general lane lines.
    3.  **Gemini Hybrid**: An advanced model where a large language model (Gemini) assists with decision-making in complex situations like intersections.
-   **Systematic Performance Analysis**: Employs a phased experimental design to systematically evaluate model performance at various speeds (30, 45, 60 km/h) and control frequencies (20Hz and 60Hz).
-   **Real-time Optimization**: Utilizes techniques like **Numba** for JIT compilation of performance-critical code and **threading** for non-blocking API calls to the AI model.
-   **Automated Experimentation**: Includes Python scripts to run experiments in batches, automating the process of data collection.
-   **Detailed Data Visualization**: Provides scripts to automatically analyze log data and generate graphs for success rates, lap times, steering stability, and trajectories.

## Technology Stack

-   **Simulator**: Webots
-   **Language**: Python 3
-   **Core Libraries**:
    -   `pandas`
    -   `numpy`
    -   `opencv-python`
    -   `matplotlib`
    -   `seaborn`
    -   `numba`
-   **AI Model**: Google Gemini API (e.g., `gemini-2.5-flash`)

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone [your-repository-url]
    cd [your-repository-name]
    ```

2.  **Install Webots**: Download and install the appropriate version of [Webots](https://cyberbotics.com/download).

3.  **Set up Python Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing the libraries above.)*

5.  **Set up API Key**:
    -   Create a file named `.env` in the root directory.
    -   Add your Google AI API key to the file like this:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

## How to Use

#### 1. Run Experiments

The batch processing script (`run_batch.py` etc.) allows you to run a series of experiments automatically.

-   **Configuration**: Before running, open the script and check that the paths to the Webots executable and the world file are correct for your environment.
-   **Execution**:
    ```bash
    python run_experiments.py
    ```
-   Log files will be generated in the `controllers/autonomous_car/logs` directory.

#### 2. Analyze Results

After the experiments are complete, run the analysis script.

-   **Configuration**: Open the analysis script (e.g., `analyze_results.py`) and ensure `LOGS_DIR` and `NUM_FILES_TO_ANALYZE` are set correctly for the experiment you want to analyze.
-   **Execution**:
    ```bash
    python analyze_results.py
    ```
-   A summary table will be printed to the console, and graph images will be saved to the `analysis_results` directory.

## Project Structure

```
/
├── controllers/
│   └── autonomous_car/
│       ├── autonomous_car.py       # Main controller orchestrator
│       ├── modes/                    # Driving logic modules
│       │   ├── base_mode.py
│       │   ├── mode_line_follow.py
│       │   └── ...
│       ├── logs/                     # Directory for log files (auto-generated)
│       └── utils/                    # Utility modules (e.g., LogManager)
├── worlds/
│   └── city.wbt                  # Webots world file
├── analysis_results/               # Directory for graph images (auto-generated)
├── run_experiments.py              # Batch experiment script
├── analyze_results.py              # Data analysis script
└── README.md                       # This file
```

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.


---

## Acknowledgements（謝辞）

This project utilizes sample assets (e.g., `city.wbt`) from the Webots software, developed by Cyberbotics Sàrl. These assets are licensed under the Apache License, Version 2.0.

このプロジェクトは、[Cyberbotics Sàrl](https://cyberbotics.com/)によって開発されたWebotsのサンプルアセット（`city.wbt`など）を利用しています。これらのアセットは、Apache License 2.0 のもとでライセンスされています。

-   **Webots:** [https://cyberbotics.com/](https://cyberbotics.com/)
-   **Apache License 2.0:** [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)
