# Multi-Pawn Quantum Maze Simulation

This repository implements a quantum-inspired reinforcement learning (RL) approach to navigate pawns through a dynamically generated maze. The simulation integrates quantum computing principles, neural networks, and reinforcement learning to showcase multi-agent coordination in a structured environment.

## Features
- **Quantum Circuit Simulation**: Utilizes Qiskit to perform quantum gate operations and generate directional moves based on qubit states.
- **Reinforcement Learning**: Implements policy optimization using PyTorch to enable adaptive learning for maze navigation.
- **Random Maze Generation**: Dynamically generates mazes using a depth-first search (DFS) algorithm.
- **Visualization with Pygame**: Displays the maze, pawns, visited paths, and decision-making process in real-time.
- **Tkinter Integration**: Provides a separate pop-out action log window for detailed analysis of pawn movements and decisions.
- **Customizable Reward System**: Adjusts rewards and penalties to encourage optimal navigation.

---

## Requirements
### Software
- Python 3.9 or higher
- Libraries:
  - `pygame`
  - `qiskit`
  - `torch`
  - `tkinter`
  - `qiskit-aer`
 
  
#### **NOTE**: This has only been tested in Python 3.11.9
---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dev-nolant/Qubits-Maze.git
   cd Qubits-Maze
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the simulation:
   ```bash
   python main.py
   ```
2. Control options:
   - `[A]`: Toggle auto-step mode.
   - `[Space]`: Perform a single step when auto-step is off.
   - `[Tab]`: Display keybinds and instructions.
   - `[P]`: Open a pop-out action log window.
   - `[ESC]`: Exit the simulation.

---

## How It Works

### Quantum Circuit Integration
Each pawn's movement is guided by the measurement outcomes of a 3-qubit quantum circuit. The circuit includes a combination of Hadamard and Identity gates applied to each qubit. The outcomes of the measurement determine the directional move of the pawn in the maze.

### Reinforcement Learning (RL)
- The neural network is trained to optimize the pawns' decision-making processes using a shared policy across all pawns.
- The reward system is designed to reinforce movement toward the exit while penalizing inefficient or incorrect paths.

### Maze Representation
- **Start (S)**: The starting point of the maze.
- **Exit (E)**: The goal of the maze.
- **Walls (1)**: Impassable regions in the maze.
- **Open Path (0)**: Navigable regions.
- **Breadcrumb Trail**: Cells visited by pawns are marked with increasing intensity of green, reflecting the number of visits.

### Visualization
The maze and pawns are displayed in real-time using Pygame, while a separate Tkinter-based pop-out window logs the detailed decisions and actions taken by the pawns.

---

## Reward System
- **Base Reward**: Each step incurs a small penalty (-0.05).
- **Wall Penalty**: Collisions with walls result in a larger penalty (-2.0).
- **Closer to Exit**: Movement closer to the exit is rewarded (+0.5).
- **Further from Exit**: Movement further from the exit is penalized (-0.2).
- **Revisiting Cells**: Revisiting recently visited cells incurs a penalty (-0.3).
- **Corner Penalty**: Moving into a dead-end corner results in a significant penalty (-2.0).
- **Exit Bonus**: Reaching the exit earns a substantial reward (+50.0).

---

## Future Work
- **Multi-agent Collaboration**: Extend the simulation to include collaborative behaviors among pawns.
- **Dynamic Maze Updates**: Introduce real-time changes to the maze structure.
- **Enhanced Quantum Gates**: Experiment with more advanced quantum gates and multi-qubit entanglement.
- **Advanced Visualization**: Implement 3D rendering for the maze environment.

---
![image](https://github.com/user-attachments/assets/ad333c86-4eee-45f1-8ec9-b85e9cca478c)


## License
This project is licensed under the MIT License. See the LICENSE file for details.

