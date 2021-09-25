## Multi-Armed Bandits
Q4: 10-armed Bandit TestBed from Sutton and Barto<br>
Q6: Epsilon Greedy policy with plots for cumulative rewards and % of optimal action taken at each step<br>
Q7: Comparison between Epsilon Greedy and UCB learning policies. We also compare these policies with optimistic action-value<br>
estimate 
## Setup

* Python 3.5+ (for type hints)<br>

Install requirements with the command:<br>
```bash
pip install -r requirements.txt
```
### Code structure

- The multi-armed bandit environment is defined in `env.py`
- Epsilon-greedy/UCB agents are given in `agents.py`. Both agents inherit the abstract class `BanditAgent`.
- To break ties randomly, we define our own argmax function in `agents.py`.
- The main file to run code is `main.py`, with functions for each question.

Please find comments for each question.<br>
## Run the code
```bash
python main.py
```
