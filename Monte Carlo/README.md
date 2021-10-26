# Exercise 4: Monte-Carlo Methods

## Setup

* Python 3.5+ (for type hints)

Install requirements with the command:
```bash
pip install -r requirements.txt
```

### Code structure

- Envs: Blackjack is contained within the `gym` library, and a custom Four Rooms gym environment is given in `env.py`. Complete the `step()` function. To use the custom environment, call `register_env()` once.
- Algorithms are given in `algorithms.py`.
- Policies for the blackjack and Four Rooms environments are given in `policy.py`. For these simple policies, we represent policies as closures so that the policies are updated as soon as the Q-values are updated.

## Run the code
```bash
python main.py
```




