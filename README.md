# 🤖 RoboGambit ♟️

Welcome to **RoboGambit**, an autonomous chess system developed for the 2025-26 Aries and Robotics Club, IIT Delhi! This project is divided into two major tasks: a high-performance 6x6 chess engine and an ArUco-based computer vision perception module.

---

## 📂 Project Structure

- 📂 **`Task 1/`**
  - 📜 [`game.py`](file:///home/amey/Desktop/ROBOGAMBIT/Task%201/game.py): The core autonomous chess engine.
- 📂 **`Task 2/`**
  - 📜 [`perception.py`](file:///home/amey/Desktop/ROBOGAMBIT/Task%202/perception.py): Computer vision module for board reconstruction.
  - 🖼️ `board_*.png`: Sample board images for testing.

---

## 🧠 Task 1: Autonomous Game Engine

The engine evaluates 6x6 board positions to determine the absolute best legal move using advanced searching and evaluation techniques.

### Core Features:
- **Minimax Algorithm** with Alpha-Beta Pruning for deep lookahead.
- **Quiescence Search** to stabilize evaluations during volatile capture sequences.
- **Piece-Square Tables (PST)** for optimized positional play.
- **Material & Structure Evaluation** to assess board advantages.

---

## 👁️ Task 2: Perception Module

The perception system uses computer vision to reconstruct the physical board state from camera images.

### Key Capabilities:
- **ArUco Detection**: Identifies corner markers (IDs 21–24) and piece markers (IDs 1–10).
- **Homography Transformation**: Computes the perspective mapping from camera pixels to world coordinates (mm).
- **Board Reconstruction**: Maps detected pieces to a 6x6 grid based on their world positions.
- **Visualization**: Real-time rendering of detected markers and the reconstructed digital board.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- NumPy (`pip install numpy`)
- OpenCV (`pip install opencv-python`)

### Running the Game Engine
To integrate or test the engine:
```python
from Task_1.game import get_best_move, initial_board
import numpy as np

# Get the best move for White
move = get_best_move(initial_board, playing_white=True)
print(f"Engine Move: {move}")
```

### Running Perception
To process a board image:
```bash
python3 Task\ 2/perception.py Task\ 2/board_1.png
```

---

## 🛠️ Tech Stack
- **Python 3** 🐍
- **NumPy**: High-speed array operations for matrix evaluations.
- **OpenCV**: Robust computer vision and ArUco marker processing.

---

*RoboGambit: Bridging physical chess with autonomous intelligence.* 🏁
