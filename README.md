# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

Name: Mukesh Kumar S

Reg No.: 212223240099

```python

import gymnasium as gym
import numpy as np

# =====================================
# Create FrozenLake Environment
# =====================================
env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped

# =====================================
# Policy Evaluation Function
# =====================================
def policy_evaluation(env, gamma=0.88, theta=1e-6):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(n_states)

    # Uniform Random Policy
    policy = np.ones((n_states, n_actions)) / n_actions

    iterations = 0

    while True:
        delta = 0

        for state in range(n_states):

            value = 0

            for action in range(n_actions):

                action_prob = policy[state][action]

                transitions = env.P[state][action]

                for prob, next_state, reward, terminated in transitions:

                    value += action_prob * prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            delta = max(delta, abs(V[state] - value))
            V[state] = value

        iterations += 1

        if delta < theta:
            break

    return V, iterations


# =====================================
# Run Policy Evaluation
# =====================================
V, iterations = policy_evaluation(env)

# =====================================
# Display Results
# =====================================
print("Number of Iterations:", iterations)

print("\nState Value Function:\n")
print(V)

print("\nState Value Function (4 x 4 Grid):\n")
print(V.reshape(4,4))

```

---

## Output




```text


Number of Iterations: 38

State Value Function:

[0.01235356 0.01042298 0.01933735 0.00947697 0.01478582 0.
 0.03889412 0.         0.0326019  0.08433739 0.13781067 0.
 0.         0.17034467 0.43357932 0.        ]

State Value Function (4 x 4 Grid):

[[0.01235356 0.01042298 0.01933735 0.00947697]
 [0.01478582 0.         0.03889412 0.        ]
 [0.0326019  0.08433739 0.13781067 0.        ]
 [0.         0.17034467 0.43357932 0.        ]]


```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text

Number of Iterations: 25

State Value Function:

[0.00361226 0.00350253 0.00880655 0.00345955 0.00569497 0.
 0.02426174 0.         0.01657953 0.05308722 0.10147413 0.
 0.         0.12325197 0.3838971  0.        ]

State Value Function (4 x 4 Grid):

[[0.00361226 0.00350253 0.00880655 0.00345955]
 [0.00569497 0.         0.02426174 0.        ]
 [0.01657953 0.05308722 0.10147413 0.        ]
 [0.         0.12325197 0.3838971  0.        ]]
​

```
---


