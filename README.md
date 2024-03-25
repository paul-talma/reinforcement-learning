# reinforcement-learning
Implementations of reinforcement learning algorithms, including Deep Q-Learning

toy_text contains warm-up reinforcement learning environments. The small state space allows classic Q-Learning to succeed in solving the environments.

classic_control contains environments with (in-principle) continuous state spaces, requiring the use of deep Q-learning methods to solve.

To run one of the agents, first run the env_main.py file, to train the agent, then run the env_testing.py file to observe the agent's performance. For example, run cartpole_main.py, then cartpole_testing.py. The main.py file saves the trained model in the corresponding saved_models folder.

The file tree is as follows:
.
├── LICENSE
├── README.md
├── classic_control
│   ├── acrobot
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── action_selection.cpython-312.pyc
│   │   │   ├── deep_q_network.cpython-312.pyc
│   │   │   ├── replay_buffer.cpython-312.pyc
│   │   │   ├── training_functions.cpython-312.pyc
│   │   │   └── visualization.cpython-312.pyc
│   │   ├── acrobot_main.py
│   │   ├── acrobot_testing.py
│   │   ├── action_selection.py
│   │   ├── deep_q_network.py
│   │   ├── replay_buffer.py
│   │   ├── saved_models
│   │   │   └── acrobot_model.pth
│   │   ├── soft_update.py
│   │   ├── training_functions.py
│   │   └── visualization.py
│   ├── cartpole
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── action_selection.cpython-312.pyc
│   │   │   ├── deep_q_network.cpython-312.pyc
│   │   │   ├── epsilon_greedy.cpython-312.pyc
│   │   │   ├── replay_buffer.cpython-312.pyc
│   │   │   ├── training.cpython-312.pyc
│   │   │   ├── training_functions.cpython-312.pyc
│   │   │   └── visualization.cpython-312.pyc
│   │   ├── action_selection.py
│   │   ├── cartpole_main.py
│   │   ├── cartpole_testing.py
│   │   ├── deep_q_network.py
│   │   ├── replay_buffer.py
│   │   ├── saved_models
│   │   │   └── cart_pole_model.pth
│   │   ├── soft_update.py
│   │   ├── training_functions.py
│   │   └── visualization.py
│   └── mountain_car
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── action_selection.cpython-312.pyc
│       │   ├── deep_q_network.cpython-312.pyc
│       │   ├── replay_buffer.cpython-312.pyc
│       │   ├── training_functions.cpython-312.pyc
│       │   └── visualization.cpython-312.pyc
│       ├── action_selection.py
│       ├── deep_q_network.py
│       ├── mountain_car_main.py
│       ├── mountain_car_testing.py
│       ├── replay_buffer.py
│       ├── saved_models
│       ├── soft_update.py
│       ├── training_functions.py
│       └── visualization.py
└── toy_text
    ├── __init__.py
    ├── __pycache__
    │   ├── params.cpython-312.pyc
    │   ├── q_agent_template.cpython-312.pyc
    │   ├── quickplot.cpython-312.pyc
    │   └── visualization.cpython-312.pyc
    ├── cliffwalking.py
    ├── frozen_lake_4x4.py
    ├── frozen_lake_8x8.py
    ├── q_agent_template.py
    ├── quickplot.py
    └── visualization.py