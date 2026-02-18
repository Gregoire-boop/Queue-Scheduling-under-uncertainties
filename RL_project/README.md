# RL project

## Run your modified model
## To make it is work, you have to modify ChildEnv.py and ChildPolicy.py
## Tou can duplicate these two files to test as many policies as you want.

```bash
python -m app.main
```

## Generate data
## Data will be generated in appd/data/data_files
```bash
python -m app.InstanceGenerator
```

## Model example: Random
## You can test the environment with the Random model
## It selects randomly 10 clients for the observation
## Then the action selects randomly 
```bash
python -m app.run_random
```

## Evaluation
## You can evaluate your model on 50 pregenerated instances
## It allows you to compare yourself
```bash
python -m app.evaluate
```