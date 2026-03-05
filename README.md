# 🤖 RoboGambit ♟️

Welcome to **RoboGambit**, an autonomous 6x6 chess engine developed for Task 1 of the 2025-26 Aries and Robotics Club, IIT Delhi!

---

## 🌟 Overview

The engine evaluates board positions from either White's or Black's perspective to determine the absolute best legal move. It uses:
- **Minimax Algorithm** with Alpha-Beta Pruning 🧠
- **Quiescence Search** to prevent the horizon effect in volatile capture scenarios ⚔️
- **Piece-Square Tables (PST)** to optimize optimal positional deployment 🗺️
- **Static Board Evaluation** covering material advantages and pawn structure weaknesses 🏗️

---

## 🛠️ Files Included

- 📜 **`ROBOGAMBIT.py`** 
  The core engine itself. It calculates piece move generation (Pawns, Knights, Bishops, Queens, and Kings), maps coordinates to indices, evaluates static board metrics, and employs minimax to predict optimal lines of play.
- 🎯 **`practice2.py`** 
  An interactive CLI test environment carrying 100 heavily verified chess positions ranging from single-move Checkmates and Free Captures to complex Endgame and High-ELO situations. *It is set up to automatically feed positions into the engine and trace its responses.*

---

## 🚀 How to Run the Test Suite

We've automated `practice2.py` so you can watch `ROBOGAMBIT.py` battle-test itself across the 100 positions!

1. **Fire up the tester:**
   ```bash
   python3 practice2.py
   ```
2. **Choose your category:**
   When prompted, choose what scenarios you want the engine to solve:
   - `[1]` All 100 scenarios 🏆
   - `[5]` Checkmates only 🪦
   - `[6]` Free captures only 🥷
3. **Watch the Engine Think:**
   The terminal will output the engine's internal evaluation score for every candidate move, then officially lock in its best move, awarding itself points if the move is legal.

---

## 💻 Tech Stack & Complexity
- Developed purely in **Python3** 🐍
- Employs **NumPy** for hyper-fast 6x6 array operations and matrix evaluations ⚡
- Optimized Move Generator logic avoids recursion lag, operating at `O(36)` scaling ⏱️

---

*RoboGambit: Checkmate the competition in 6x6 spaces.* 🏁
