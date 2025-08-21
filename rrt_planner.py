import numpy as np
import time
import yaml
from scipy.spatial import KDTree
from multi_drone import MultiDrone

class rrt_planner:
    def __init__(self, sim, max_iter=30000, base_step_size=2.0, smooth_iterations=100):
        """
        Advanced RRT-Connect with three key optimizations:
        1. Gaussian Sampling around bridge regions
        2. Leader-follower guided sampling
        3. Adaptive step size
        """
        self.sim = sim
        self.max_iter = max_iter
        self.smooth_iterations = smooth_iterations
        self.num_drones = sim.N
        self.environment_file = getattr(sim, 'environment_file', 'environment.yaml') # Safely get env file

        # Adaptive step size parameters
        self.step_size = base_step_size
        self.min_step = base_step_size * 0.2
        self.max_step = base_step_size * 2.0
        self.step_increase_rate = 1.05
        self.step_decrease_rate = 0.8

        # Gaussian sampling parameters
        self.gaussian_sampling_enabled = False
        self.gaussian_center = None
        self.gaussian_std = base_step_size * 2.0
        self.bridge_update_frequency = 100
        self.gaussian_sample_ratio = 0.35 # Slightly increase for complex env

        # Leader-follower parameters
        self.leader_path = None
        self.leader_sample_ratio = 0.25 # Slightly increase for high-dim
        self.leader_dispersion = base_step_size * 1.5

        # Goal biasing
        self.goal_bias = min(0.05 + 0.02 * self.num_drones, 0.20)

        self.q_init = self.sim.initial_configuration
        self.q_goal = self.sim.goal_positions

        # Initialize trees
        self.tree_a = {'nodes': [self.q_init], 'parent': {0: None}, 'kdtree': None, 'kdtree_outdated': True}
        self.tree_b = {'nodes': [self.q_goal], 'parent': {0: None}, 'kdtree': None, 'kdtree_outdated': True}

        # Performance tracking
        self.stats = {k: 0 for k in ['gaussian_samples', 'leader_samples', 'random_samples', 'goal_samples',
                                     'step_adjustments', 'successful_extends', 'failed_extends']}

        self._compute_leader_path()

    def _compute_leader_path(self):
        """
        Strategy 2: More robustly compute a guide path for a single leader drone.
        This version creates a temporary single-drone config to avoid assertion errors.
        """
        print("Computing leader path...")
        if self.num_drones <= 0: return

        leader_idx = self.num_drones // 2
        
        try:
            # Load the full environment config from file
            with open(self.environment_file, 'r') as f:
                full_config = yaml.safe_load(f)

            # Create a new config specifically for the single leader drone
            leader_config = {
                'bounds': full_config['bounds'],
                'obstacles': full_config.get('obstacles', []),
                'initial_configuration': [self.q_init[leader_idx].tolist()],
                'goals': [full_config['goals'][leader_idx]]
            }
            
            # Create a temporary single-drone simulation using the new config dictionary
            temp_sim = MultiDrone(num_drones=1, environment_config=leader_config)
            
            # Use a simplified but complete RRT planner for the leader
            # Give it more time and iterations to solve complex mazes
            simple_planner = rrt_planner_simple(temp_sim, max_iter=5000) 
            path = simple_planner.plan(timeout=20) # Give it up to 20 seconds

            if path:
                # Extract the 3D coordinates from the path of configurations
                self.leader_path = np.array([config[0] for config in path])
                print(f"Leader path found with {len(self.leader_path)} waypoints.")
            else:
                # Fallback if leader path is still not found
                print("Leader path not found, using straight line fallback.")
                self.leader_path = np.array([self.q_init[leader_idx], self.q_goal[leader_idx]])
        except Exception as e:
            print(f"Error computing leader path: {e}. Using fallback.")
            self.leader_path = np.array([self.q_init[leader_idx], self.q_goal[leader_idx]])
    
    def _update_bridge_region(self):
        if len(self.tree_a['nodes']) < 5 or len(self.tree_b['nodes']) < 5: return
        sample_size = min(30, len(self.tree_a['nodes']), len(self.tree_b['nodes']))
        indices_a = np.random.choice(len(self.tree_a['nodes']), sample_size, replace=False)
        indices_b = np.random.choice(len(self.tree_b['nodes']), sample_size, replace=False)
        nodes_a = np.array([self.tree_a['nodes'][i] for i in indices_a])
        nodes_b = np.array([self.tree_b['nodes'][i] for i in indices_b])
        dists = np.linalg.norm(nodes_a[:, np.newaxis, :, :] - nodes_b[np.newaxis, :, :, :], axis=(2, 3))
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[idx]
        self.gaussian_center = (nodes_a[idx[0]] + nodes_b[idx[1]]) / 2
        self.gaussian_std = min_dist / 3.0
        self.gaussian_sampling_enabled = True

    def _gaussian_sample(self):
        if not self.gaussian_sampling_enabled: return self._random_sample()
        self.stats['gaussian_samples'] += 1
        sample = np.random.normal(loc=self.gaussian_center, scale=self.gaussian_std)
        return np.clip(sample, self.sim._bounds[:, 0], self.sim._bounds[:, 1])

    def _leader_guided_sample(self):
        if self.leader_path is None or len(self.leader_path) < 2: return self._random_sample()
        self.stats['leader_samples'] += 1
        t = np.random.rand() * (len(self.leader_path) - 1)
        idx = int(t)
        alpha = t - idx
        leader_pos = (1 - alpha) * self.leader_path[idx] + alpha * self.leader_path[idx + 1]
        offsets = np.random.randn(self.num_drones, 3) * self.leader_dispersion
        config = leader_pos + offsets
        return np.clip(config, self.sim._bounds[:, 0], self.sim._bounds[:, 1])

    def _random_sample(self):
        self.stats['random_samples'] += 1
        return np.random.uniform(self.sim._bounds[:, 0], self.sim._bounds[:, 1], (self.num_drones, 3))

    def _intelligent_sample(self):
        rand = np.random.random()
        if rand < self.goal_bias:
            self.stats['goal_samples'] += 1
            return self.q_goal
        elif rand < self.goal_bias + self.leader_sample_ratio:
            return self._leader_guided_sample()
        elif rand < self.goal_bias + self.leader_sample_ratio + self.gaussian_sample_ratio:
            return self._gaussian_sample()
        else:
            return self._random_sample()

    def _adaptive_extend(self, tree, q_target):
        nearest_idx = self._find_nearest(tree, q_target)
        if nearest_idx is None: return None, None
        q_near = tree['nodes'][nearest_idx]
        direction = q_target - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return None, None
        q_new = q_near + direction / dist * self.step_size if dist > self.step_size else q_target
        if self.sim.motion_valid(q_near, q_new):
            self.step_size = min(self.step_size * self.step_increase_rate, self.max_step)
            self.stats['successful_extends'] += 1
            new_idx = len(tree['nodes'])
            tree['nodes'].append(q_new)
            tree['parent'][new_idx] = nearest_idx
            tree['kdtree_outdated'] = True
            return new_idx, q_new
        else:
            self.step_size = max(self.step_size * self.step_decrease_rate, self.min_step)
            self.stats['failed_extends'] += 1
            return None, None

    def _adaptive_connect(self, tree, q_target):
        last_idx, last_q = -1, None
        for _ in range(30):
            status_idx, status_q = self._adaptive_extend(tree, q_target)
            if status_q is None: return "Trapped", last_idx if last_idx != -1 else None
            last_idx, last_q = status_idx, status_q
            if np.linalg.norm(status_q - q_target) < 1e-6: return "Reached", status_idx
        return "Advanced", last_idx

    def plan(self, timeout=110):
        start_time = time.time()
        if not self.sim.is_valid(self.q_init) or not self.sim.is_valid(self.q_goal): return None
        if self.sim.motion_valid(self.q_init, self.q_goal): return [self.q_init, self.q_goal]

        for i in range(1, self.max_iter + 1):
            if time.time() - start_time > timeout: break
            if i % 10 == 0:
                self._update_kdtree(self.tree_a)
                self._update_kdtree(self.tree_b)
            if i % self.bridge_update_frequency == 0: self._update_bridge_region()

            q_rand = self._intelligent_sample()
            q_new_idx, q_new = self._adaptive_extend(self.tree_a, q_rand)

            if q_new is not None:
                connect_result, q_b_idx = self._adaptive_connect(self.tree_b, q_new)
                if connect_result != "Trapped" and q_b_idx is not None:
                    q_conn = self.tree_b['nodes'][q_b_idx]
                    if np.linalg.norm(q_new - q_conn) < self.step_size * 2:
                        final_path = self._reconstruct_path(self.tree_a, self.tree_b, q_new_idx, q_b_idx)
                        if final_path:
                           return self._smooth_path(final_path)

            self._swap_trees()
        return None

    def _update_kdtree(self, tree):
        if len(tree['nodes']) > 0 and tree['kdtree_outdated']:
            tree['kdtree'] = KDTree(np.array(tree['nodes']).reshape(len(tree['nodes']), -1))
            tree['kdtree_outdated'] = False

    def _find_nearest(self, tree, q_target):
        self._update_kdtree(tree)
        if tree['kdtree']:
            _, idx = tree['kdtree'].query(q_target.flatten())
            return idx
        return 0

    def _swap_trees(self):
        self.tree_a, self.tree_b = self.tree_b, self.tree_a

    def _reconstruct_path(self, tree1, tree2, idx1, idx2):
        path1, curr = [], idx1
        while curr is not None: path1.append(tree1['nodes'][curr]); curr = tree1['parent'].get(curr)
        path1.reverse()
        
        path2, curr = [], idx2
        while curr is not None: path2.append(tree2['nodes'][curr]); curr = tree2['parent'].get(curr)
        path2.reverse()
        
        # Combine paths intelligently
        combined_path = path1
        if np.linalg.norm(path1[-1] - path2[0]) > 1e-5: # If they didn't meet at the same node
            combined_path.append(path2[0])
        combined_path.extend(path2[1:])

        # Final check for correct start
        if np.array_equal(tree1['nodes'][0], self.q_init): return combined_path
        else: return combined_path[::-1]


    def _smooth_path(self, path, max_iterations=None):
        max_iterations = max_iterations or self.smooth_iterations
        if len(path) <= 2: return path
        smoothed = path.copy()
        for _ in range(max_iterations):
            if len(smoothed) <= 2: break
            i, j = sorted(np.random.choice(len(smoothed), 2, replace=False))
            if j <= i + 1: continue
            if self.sim.motion_valid(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed


# A simplified RRT-Connect specifically for the single leader drone
class rrt_planner_simple:
    def __init__(self, sim, max_iter=5000):
        self.sim = sim
        self.max_iter = max_iter
        self.q_init = sim.initial_configuration
        self.q_goal = sim.goal_positions
        self.step_size = 2.0
        self.tree_a = {'nodes': [self.q_init], 'parent': {0: None}}
        self.tree_b = {'nodes': [self.q_goal], 'parent': {0: None}}

    def plan(self, timeout=20):
        start_time = time.time()
        for i in range(self.max_iter):
            if time.time() - start_time > timeout: return None
            
            q_rand = self.q_goal if np.random.rand() < 0.2 else np.random.uniform(self.sim._bounds[:, 0], self.sim._bounds[:, 1], (1, 3))
            
            if not self._extend(self.tree_a, q_rand):
                self._swap()
                continue
            
            q_new = self.tree_a['nodes'][-1]
            if self._connect(self.tree_b, q_new):
                return self._reconstruct()

            self._swap()
        return None

    def _extend(self, tree, target):
        nodes_arr = np.array(tree['nodes']).squeeze(axis=1)
        dists = np.linalg.norm(nodes_arr - target, axis=1)
        near_idx = np.argmin(dists)
        q_near = tree['nodes'][near_idx]
        
        direction = target - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return False

        q_new = q_near + direction/dist * self.step_size if dist > self.step_size else target
        
        if self.sim.motion_valid(q_near, q_new):
            tree['nodes'].append(q_new)
            tree['parent'][len(tree['nodes'])-1] = near_idx
            return True
        return False

    def _connect(self, tree, target):
        while True:
            prev_node_count = len(tree['nodes'])
            if self._extend(tree, target):
                if np.linalg.norm(tree['nodes'][-1] - target) < 1e-5: return True
            else:
                return len(tree['nodes']) > prev_node_count
        return False

    def _reconstruct(self):
        # Find connection point by checking the last nodes added
        last_node_a = self.tree_a['nodes'][-1]
        last_node_b = self.tree_b['nodes'][-1]

        # The connection point is the target of the successful connect call
        # which is the last node of the other tree
        if np.linalg.norm(last_node_a - last_node_b) > 1e-5:
            # This case happens if connect advanced but didn't reach
             if len(self.tree_a['nodes']) > len(self.tree_b['nodes']): # Tree_a was growing
                 q_b = self.tree_b['nodes'][self._find_nearest_simple(self.tree_b, last_node_a)]
                 self.tree_a['nodes'].append(q_b) # Manually connect
             else: # Tree_b was growing
                 q_a = self.tree_a['nodes'][self._find_nearest_simple(self.tree_a, last_node_b)]
                 self.tree_b['nodes'].append(q_a)
        
        path1, curr = [], len(self.tree_a['nodes']) - 1
        while curr is not None: path1.append(self.tree_a['nodes'][curr]); curr = self.tree_a['parent'].get(curr)
        path1.reverse()
        
        path2, curr = [], len(self.tree_b['nodes']) - 1
        while curr is not None: path2.append(self.tree_b['nodes'][curr]); curr = self.tree_b['parent'].get(curr)
        path2.reverse()

        # Reconcile connection
        if np.linalg.norm(path1[-1] - path2[-1]) < 1e-5: # They met
            path2.pop(-1)
        
        path2.reverse()
        return path1 + path2

    def _find_nearest_simple(self, tree, target):
        dists = np.linalg.norm(np.array(tree['nodes']).squeeze(axis=1) - target, axis=1)
        return np.argmin(dists)

    def _swap(self): self.tree_a, self.tree_b = self.tree_b, self.tree_a

def run_experiment(exp_name, configs, num_runs=20):
    results = {}
    
    for config in configs:
        env_file = config['env_file']
        num_drones = config['num_drones']
        
        print(f"--- Running Experiment: {exp_name} | Env: {env_file}, Drones: {num_drones} ---")
        
        run_times, success_count = [], 0
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end="", flush=True)
            try:
                sim = MultiDrone(num_drones=num_drones, environment_file=env_file)
                # This attribute is used by the planner to find the file
                sim.environment_file = env_file
                planner = rrt_planner(sim)
                
                start_time = time.time()
                path = planner.plan(timeout=110)
                end_time = time.time()
                
                if path:
                    run_times.append(end_time - start_time)
                    success_count += 1
                    print(f" Success ({end_time - start_time:.2f}s)")
                else:
                    print(" Failure")
            except Exception as e:
                print(f" ERROR: {e}")
        
        key = f"Env: {env_file}, Drones: {num_drones}"
        results[key] = {'run_times': run_times, 'success_rate': (success_count/num_runs)*100, 'successes': success_count}

    print(f"\n{'='*20} Experiment Summary: {exp_name} {'='*20}")
    for key, data in results.items():
        print(f"\nConfiguration: {key}")
        print(f"Success Rate: {data['success_rate']:.1f}% ({data['successes']}/{num_runs})")
        if data['run_times']:
            mean_time, std_dev = np.mean(data['run_times']), np.std(data['run_times'])
            ci = 1.96 * std_dev / np.sqrt(len(data['run_times'])) if len(data['run_times']) > 0 else 0
            print(f"Average Runtime (on success): {mean_time:.4f} seconds")
            print(f"Standard Deviation: {std_dev:.4f}")
            print(f"95% Confidence Interval: +/- {ci:.4f} seconds")
        else:
            print("No successful runs for runtime analysis.")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    
    # --- Experiment for Question B4: Environmental Complexity ---
    exp_B4_configs = [
        {'env_file': 'env_lv1.yaml', 'num_drones': 3},
        {'env_file': 'env_lv2.yaml', 'num_drones': 3},
        {'env_file': 'env_lv3.yaml', 'num_drones': 3},
        {'env_file': 'env_lv5.yaml', 'num_drones': 6},
    ]
    # To save time, you can comment out the B4 experiment while testing B5
    run_experiment("Environmental Complexity", exp_B4_configs, num_runs=20)

    # --- Experiment for Question B5: Number of Drones (Curse of Dimensionality) ---
    # UPDATED to use the modified, easier environment files
    exp_B5_configs = [
        {'env_file': 'env_lv5_k4_mod.yaml', 'num_drones': 4},
        {'env_file': 'env_lv5_k8_mod.yaml', 'num_drones': 8},
        {'env_file': 'env_lv5_k12_mod.yaml', 'num_drones': 12},
    ]
    run_experiment("Curse of Dimensionality", exp_B5_configs, num_runs=10) # Using 10 runs for B5 is reasonable

    print("All experiments completed.")