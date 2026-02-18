from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
from gymnasium import spaces
import numpy as np

class ChildEnv(Env):

    K = 500

    def _get_action_space(self):
        # Actions 0 à K-1 : Clients dans la file
        # Action K : HOLD
        return spaces.Discrete(self.K + 1)
    
    def _get_observation_space(self):
        return spaces.Dict({
            "waiting_customers": spaces.Box(low=-1, high=np.inf, shape=(self.K, 4), dtype=np.float32),
            "server_status": spaces.Box(low=0, high=np.inf, shape=(self.c, 2), dtype=np.float32),
            "context": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        })
    
    def _get_obs(self):
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        
        queue_matrix = np.full((self.K, 4), -1.0, dtype=np.float32)
        sorted_customers = sorted(waiting_customers.values(), key=lambda c: sim_time - c.arrival_time)
        
        for i, customer in enumerate(sorted_customers):
            if i >= self.K: break
            
            is_appointment = 1.0 if customer.id in appointments else 0.0
            abandonment_info = -1.0 if customer.abandonment_time is None else customer.abandonment_time

            avg_duration = np.mean([server.avg_service_time[customer.task] for server in self.servers.values()])
            normalized_arrival = customer.arrival_time / 1440.0
            normalized_avg_duration = avg_duration / 60.0

            queue_matrix[i] = [
                normalized_arrival,
                normalized_avg_duration, 
                is_appointment,
                abandonment_info
            ]

        server_matrix = np.zeros((self.c, 2), dtype=np.float32)
        for s_id in range(self.c):
            end_time = expected_end.get(s_id, 0.0)
            server_matrix[s_id][0] = end_time
            server_matrix[s_id][1] = 1.0 if s_id == selected_server_id else 0.0

        context_array = np.array([sim_time, len(waiting_customers)], dtype=np.float32)

        return {
            "waiting_customers": queue_matrix,
            "server_status": server_matrix,
            "context": context_array
        }
    
    def _get_customer_from_action(self, action) -> Customer:
        if action == self._get_hold_action_number():
            return None
        
        sorted_customers = sorted(self.customer_waiting.values(), key=lambda c: c.arrival_time)
        if 0 <= action < len(sorted_customers):
            return sorted_customers[action]
        return None  

    def _get_invalid_action_reward(self) -> float: 
        return -100.0
    
    def _get_valid_reward(self, customer):
        reward = 0.0
        current_time = self.system_time
        
        # appointment reward
        if customer.id in self.appointments:
            appt_time = self.appointments[customer.id].time
            delta_min = appt_time - current_time # pos if early, neg if late
            
            MAX_APPT_BONUS = 50.0 # max reward
            
            if delta_min > 60:
                # too early, no recompense
                reward_appt = 0.0
                
            elif 3 < delta_min <= 60:
                # early, decrease linearly from max reward at 3 min early to 0 reward at 60 min early
                factor = 1.0 - ((delta_min - 3) / 57.0)
                reward_appt = MAX_APPT_BONUS * factor
                
            elif -3 <= delta_min <= 3:
                reward_appt = MAX_APPT_BONUS
                
            else:
                # late, decrease linearly from max reward at 3 min late to 0 reward at 53 min late, then -10 reward at 60 min late and beyond
                late_min = abs(delta_min + 5)
                # 2 points lost per minute late, with a grace period of 5 minutes after the appointment time
                reward_appt = MAX_APPT_BONUS - (late_min * 2.0)
                
            reward += reward_appt

        # non-appointment reward
        else:
            reward += 20.0 # reward for serving a non-appointment customer

        wait_time = current_time - customer.arrival_time
        # lost 0.5 point per minute of waiting
        reward -= (wait_time * 0.5)

        time_left = customer.abandonment_time - current_time
        if 0 < time_left < 10.0:
            reward += 10.0

        return reward
    
    def action_masks(self):
        mask = np.zeros(self.K + 1, dtype=bool)
        sorted_customers = sorted(self.customer_waiting.values(), key=lambda c: c.arrival_time)
        current_time = self.system_time
        
        urgent_appt_indices = []
        for i, customer in enumerate(sorted_customers):
            if i >= self.K: break
            
            if customer.id in self.appointments:
                appt_time = self.appointments[customer.id].time
                if current_time >= (appt_time - 1.0):
                    urgent_appt_indices.append(i)

        if urgent_appt_indices:
            for idx in urgent_appt_indices:
                mask[idx] = True
        else:
            for i in range(min(len(sorted_customers), self.K)):
                mask[i] = True

        mask[self._get_hold_action_number()] = True
        return mask
    
    def _get_hold_action_number(self):
        return self.K