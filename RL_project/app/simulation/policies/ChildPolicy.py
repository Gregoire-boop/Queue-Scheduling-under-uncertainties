from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch as th

class ChildPolicy(Policy):
    def __init__(self, model_title):
        super().__init__(model_title)
        self.model = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        
        self.model_path = os.path.join(root_dir, "app", "data", "models", model_title)
        
        print(f"[DEBUG] Chemin modèle attendu : {self.model_path}.zip")

    def _mask_fn(self, env):
        return env.unwrapped.action_masks()
        

    def _predict(self, obs, info):
        """
        Prédit la prochaine action.
        """
        if self.model is None:
            import os
            
            parent_dir = os.path.dirname(self.model_path)
            
            print(f"\n[DEBUG] Python cherche dans : {parent_dir}")
            
            try:
                files = os.listdir(parent_dir)
                print(f"[DEBUG] Fichiers trouvés par Python : {files}")
                
                target_file = os.path.basename(self.model_path) + ".zip"
                if target_file in files:
                    print(f"[DEBUG] fichier {target_file} trouvé.")
                else:
                    print(f"[DEBUG] fichier  {target_file} non trouvé.")
            except Exception as e:
                print(f"[DEBUG] Erreur d'accès au dossier : {e}")

            print(f"[DEBUG] Tentative de chargement brut de : {self.model_path}")
            self.model = MaskablePPO.load(self.model_path)
        try :
            action_masks = self.env.env_method("action_masks")[0]
        except AttributeError:
            action_masks = self.env.unwrapped.action_masks()
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return int(action)
    
    def learn(self, scenario, total_timesteps, verbose):
        env = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        env = ActionMasker(env, self._mask_fn)
        policy_kwargs = dict(net_arch=[256, 256],
                             activation_fn=th.nn.Tanh)
        self.model = MaskablePPO(
            "MultiInputPolicy", 
            env,
            verbose=1 if verbose else 0,
            gamma=0.995,
            learning_rate=3e-4,
            ent_coef=0.1,
            batch_size=256,
            n_steps=4096,
            policy_kwargs=policy_kwargs
        )

        if verbose:
            print(f"Démarrage de l'entraînement PPO pour {total_timesteps} steps...")
        
        self.model.learn(total_timesteps=total_timesteps)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        if verbose:
            print(f"Modèle sauvegardé : {self.model_path}")