

# 🚆 Subway Surfers–Style Reinforcement Learning (Real-Time Learning)

A **2D Subway Surfers–like game** where an **AI agent learns in real time** to avoid obstacles using **Reinforcement Learning with sensors**.

This project is designed for **beginners in RL**, but follows **real-world concepts** used in robotics and autonomous systems.

---

## 🎯 Project Goal

* Build a **simple 2D game**
* Train an AI agent using **Reinforcement Learning**
* Visualize **learning live on screen**
* Agent **fails, crashes, retries, and improves**
* Use **sensor-based perception** (like real autonomous systems)

---

## 🧠 Key Idea

Instead of predicting randomly, the agent uses **3 virtual sensors**:

```
[ Left Sensor | Front Sensor | Right Sensor ]
```

Each sensor detects whether an obstacle is approaching in that direction.

This is similar to:

* Self-driving car sensors
* Robot obstacle avoidance
* Industrial control systems

---

## 🎮 How the Game Works

* The player (green block) is at the bottom
* Obstacles (red blocks) fall from the top
* The agent can move:

  * Left
  * Stay
  * Right
* The agent receives:

  * ✅ +1 reward for surviving
  * ❌ −100 reward for crashing

Over time, the agent **learns to avoid obstacles**.

---

## 🤖 Reinforcement Learning Details

* **Algorithm**: Q-Learning (tabular)
* **State Space**:
  `(left_sensor, front_sensor, right_sensor)`
* **Action Space**:
  `LEFT, STAY, RIGHT`
* **Exploration**: ε-greedy (epsilon decay)
* **Learning is visible in real time**

This is **true reinforcement learning**, not scripted behavior.

---

## 🖥️ Requirements

* Python 3.8+
* Linux / Windows / macOS

### Install dependencies:

```bash
pip install pygame numpy
```

---

## ▶️ How to Run

```bash
python subway_sensor_rl.py
```

Close the window to stop training.

---

## 👀 What You Will Observe

### Early Episodes

* Random movement
* Frequent crashes
* Low scores

### Later Episodes

* Agent avoids blocked lanes
* Moves *before* collision
* Scores increase
* Behavior looks **intentional**

➡️ You are **watching learning happen live**.

---

## 📂 Project Structure

```
.
├── subway_sensor_rl.py   # Main RL + game script
├── README.md             # Project documentation
```

(Simple by design.)

---

## 🚀 Why This Project Matters

* Beginner-friendly RL
* Real-world inspired (sensor-based decision making)
* No heavy physics engines
* No datasets
* No fake learning
* Perfect stepping stone to:

  * Deep Q-Networks (DQN)
  * Robotics RL
  * Autonomous driving simulations

---

## 🔮 Future Improvements

* Replace Q-table with **Deep Q-Network (DQN)**
* Continuous distance sensors
* Add jump / slide actions
* Increase game difficulty over time
* Save and load trained models

---

## 📌 Author

Built as part of a **learning journey into Reinforcement Learning**, focusing on:

* Simplicity
* Visualization
* Real-world relevance

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and experiment with the code.
This project is meant to be **modified, broken, and improved**.

---

