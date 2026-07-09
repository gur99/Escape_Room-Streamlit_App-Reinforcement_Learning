# RL Escape Room

`RL Escape Room` is a final project for a Reinforcement Learning course. The goal
is to build a multi-room escape game that demonstrates how different RL
algorithms learn, behave, and improve under increasing difficulty.

This repository is currently in **Stage 1**. The project skeleton is ready, the
Streamlit application starts successfully, and the codebase is organized for
gradual implementation. Full learning logic will be added room by room in later
stages.

## Project Goals

- Build an educational Escape Room experience with five rooms.
- Use a different RL method in each room.
- Visualize training behavior with rewards, episode lengths, success rate,
  exploration, and loss when relevant.
- Keep the code simple, modular, and readable for students learning RL for the
  first time.
- Make the project easy to publish on GitHub and Streamlit Cloud.

## Planned Rooms

### Room 1: Dynamic Programming

- Environment type: 10x10 grid
- Algorithm: Value Iteration or Policy Iteration
- State space: discrete grid locations
- Action space: four movement actions
- Reward function: goal reward, step penalty, and trap penalty

### Room 2: SARSA

- Environment type: 10x10 grid
- Algorithm: SARSA
- State space: discrete grid locations
- Action space: four movement actions
- Reward function: goal reward, step penalty, trap penalty, and small rewards

### Room 3: Q-Learning

- Environment type: 10x10 grid
- Algorithm: Q-Learning
- State space: discrete grid locations
- Action space: four movement actions
- Reward function: goal reward, step penalty, trap penalty, and risk-reward path design

### Room 4: Continuous Control with DQN

- Environment type: continuous 10x10 room
- Algorithm: DQN with PyTorch
- State space: `(x, y, vx, vy)`
- Action space: nine velocity-direction choices
- Reward function: fast arrival reward, step penalty, and efficiency incentive

### Room 5: Obstacle Avoidance (Optional)

- Environment type: continuous room with randomized obstacles
- Algorithm: approximate RL / deep RL extension
- State space: `(x, y, vx, vy)` with obstacle-awareness metadata
- Action space: nine velocity-direction choices
- Reward function: goal reward, progress reward, step penalty, and collision penalty

## Stage 1 Contents

Stage 1 creates the initial project skeleton only:

- modular directory structure
- Streamlit entry point
- Gymnasium-style environment interfaces
- room-specific environment skeletons
- algorithm placeholder classes
- utility modules for metrics, plotting, and replay
- `saved_runs/` directory for future experiment outputs

The current app is a **UI shell**. It displays room information and placeholder
controls, but it does not train agents yet.

## Project Structure

```text
rl_escape_room/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── algorithms/
├── environments/
├── utils/
└── saved_runs/
```

## Environment Interface

All environments are designed to follow a Gymnasium-style API:

- `reset() -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render()`
- `close()`
- `observation_space`
- `action_space`

This separation allows environments and algorithms to evolve independently.

## Hyperparameters

The Stage 1 UI already reserves space for:

- environment parameters
- algorithm hyperparameters
- future training controls

These controls are visible in the Streamlit sidebar and will become functional
as each room is implemented.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Current Status

- Stage 1 completed: project skeleton
- Next recommended step: implement **Room 1 + Dynamic Programming**
- Future stages: add training loops, plots, saved runs, comparisons, and replay
