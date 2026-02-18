from app.data.Instance import Instance
from app.simulation.envs.Env import Env
import gymnasium as gym
from app.simulation.policies.PolicyEvaluation import PolicyEvaluation
from app.simulation.policies.ChildPolicy import ChildPolicy
import pandas as pd
import os
from gymnasium.envs.registration import register

# Configuration des chemins
root_path = "app/data/data_files"
results_path = "app/data/results/tmp/"
os.makedirs(results_path, exist_ok=True)

instance_set = range(50) # Adaptez selon le nombre de fichiers générés

score = 0
NB_RUNS_INST = 1
is_valid = True

# Chargement du modèle
print("Chargement du modèle...")
model = ChildPolicy("PPO_Queue_Masked_v1") ## TODO loading

## TODO replace with your env
register(
    id="Child_Env",
    entry_point="app.simulation.envs.ChildEnv:ChildEnv", 
)
env_id = "Child_Env" 


def check_solution(instance, solution_path):
    is_valid, error = True, None
    # Tentative de lecture avec ; (votre snippet) ou , (standard)
    try:
        df = pd.read_csv(solution_path, sep=";")
        if "client" not in df.columns:
            df = pd.read_csv(solution_path, sep=",")
    except:
        return False, "Could not read CSV"

    eps = 1e-4

    # Check if the customer is served after arrival
    if (df["start"] < df["arrival"]).any():
            invalid_start = df[df["start"] < df["arrival"]]
            return False, f"Some rows have start before arrival:\n{invalid_start}"

    # Check if customer served more than once
    if df["client"].duplicated().any():
        duplicates = df[df.duplicated(subset="client", keep=False)]
        return False, f"Some clients appear more than once:\n{duplicates}"
    
    # Check the end is start + real_proc_time
    if ((df["start"] + df["real_proc_time"] - df["end"]).abs() > eps).any():
        invalid_end = df[
            (df["start"] + df["real_proc_time"] - df["end"]).abs() > eps
        ]
        return False, f"Some service have incoherent end:\n{invalid_end}"
        
    # Server and customer creation from insatnce
    customers = Env._create_customers_from_steps(instance.timeline)
    
    # Adaptation: _build_servers est une méthode d'instance, on récupère les IDs via la matrice
    # servers = Env._build_servers_from_average_matrix(instance.average_matrix)
    server_ids = set(range(len(instance.average_matrix)))
    customer_ids = set(customers.keys())
    # server_ids = set(servers.keys())

    # Invalid customer id
    if not set(df["client"]).issubset(customer_ids):
        bad = set(df["client"]) - customer_ids
        return False, f"Invalid client ids: {bad}"

    # Invalid server id
    if not set(df["server"]).issubset(server_ids):
        bad = set(df["server"]) - server_ids
        return False, f"Invalid server ids: {bad}"
    
    df = df.copy()

    # Check arrival time
    df["expected_arrival"] = df["client"].map(
        {cid: c.arrival_time for cid, c in customers.items()}
    )

    # Ajout d'une tolérance eps pour les flottants
    if not (abs(df["arrival"] - df["expected_arrival"]) < eps).all():
        bad = df[abs(df["arrival"] - df["expected_arrival"]) > eps]
        return False, f"Invalid arrival time:\n{bad}"

    # Check real service duration
    df["expected_service"] = df.apply(
        lambda r: customers[r.client].real_service_times[int(r.server)],
        axis=1
    )

    if ((df["expected_service"] - df["real_proc_time"]).abs() > eps).any():
        bad = df[
            (df["expected_service"] - df["real_proc_time"]).abs() > eps
        ]
        return False, f"Invalid service duration:\n{bad}"
        
    # Check servers serves only one customer at once
    df = df.sort_values(["server", "start"])
    # Ajout d'une tolérance eps pour éviter les faux positifs sur les chevauchements flottants
    overlap = df["start"] < (df.groupby("server")["end"].shift() - eps)

    if overlap.any():
        bad = df[overlap]
        return False, f"Server overlap detected:\n{bad[['server', 'start', 'end']]}"
    return is_valid, error

# Boucle d'évaluation
count_eval = 0
for instance_id in instance_set: 
    
    # Vérification fichier existant
    if not os.path.exists(f"{root_path}/timeline_{instance_id}.json"):
        continue

    instance = Instance.create(Instance.SourceType.FILE,
                f"{root_path}/timeline_{instance_id}.json",
                f"{root_path}/average_matrix_{instance_id}.json",
                f"{root_path}/appointments_{instance_id}.json",
                f"{root_path}/unavailability_{instance_id}.json")
    
    for run in range(NB_RUNS_INST):     
        print(f"\nEvaluation of instance {instance_id} and run {run}")   
        env = gym.make(env_id, mode=Env.MODE.TEST, instance=instance)
        sol_file = f"result_tmp_{instance_id}_{run}.csv"
        
        # Simulation
        model.simulate(env, print_logs=False, save_to_csv=True, path=results_path, file_name=sol_file)
        
        # Calcul du score métier
        policy_evaluation = PolicyEvaluation(instance.timeline, instance.appointments, clients_history=model.customers_history)
        policy_evaluation.evaluate()
        
        # Vérification technique
        is_valid, error = check_solution(instance, results_path + sol_file)
        if not is_valid:
            score = -1
            print(f"Error in instance {instance_id}, {error}")
            break

        score += policy_evaluation.final_grade
        count_eval += 1
    
    if not is_valid:
        break

if is_valid and count_eval > 0:    
    score /= count_eval

print(f"\nFinal score: {score}")