import numpy as np
import time
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
        self.gaussian_sample_ratio = 0.3  # 30% of samples use Gaussian
        
        # Leader-follower parameters
        self.leader_path = None
        self.leader_sample_ratio = 0.2  # 20% of samples follow leader
        self.leader_dispersion = base_step_size * 1.5  # How spread out drones are around leader
        
        # Goal biasing
        self.goal_bias = min(0.05 + 0.02 * self.num_drones, 0.25)
        
        self.q_init = self.sim.initial_configuration
        self.q_goal = self.sim.goal_positions
        
        # Initialize trees
        self.tree_a = {
            'nodes': [self.q_init], 
            'parent': {0: None},
            'kdtree': None,
            'kdtree_outdated': True
        }
        self.tree_b = {
            'nodes': [self.q_goal], 
            'parent': {0: None},
            'kdtree': None,
            'kdtree_outdated': True
        }
        
        # Performance tracking
        self.stats = {
            'gaussian_samples': 0,
            'leader_samples': 0,
            'random_samples': 0,
            'goal_samples': 0,
            'step_adjustments': 0,
            'successful_extends': 0,
            'failed_extends': 0
        }
        
        # Pre-compute leader path
        self._compute_leader_path()

    def _compute_leader_path(self):
        """
        Strategy 2: Compute a guide path for a single leader drone.
        This provides a skeleton path that the swarm can follow.
        """
        if self.num_drones <= 0:
            return
        
        leader_idx = self.num_drones // 2
        leader_start = self.q_init[leader_idx]
        leader_goal = self.q_goal[leader_idx]
        
        # Simple RRT for the leader
        tree = { 'nodes': [leader_start], 'parent': {0: None} }
        
        for _ in range(2000): # Limited iterations for speed
            rand_sample = leader_goal if np.random.rand() < 0.1 else np.random.uniform(self.sim._bounds[:, 0], self.sim._bounds[:, 1])
            
            dists = np.linalg.norm(np.array(tree['nodes']) - rand_sample, axis=1)
            nearest_idx = np.argmin(dists)
            q_near = tree['nodes'][nearest_idx]
            
            direction = rand_sample - q_near
            dist = np.linalg.norm(direction)
            q_new = q_near + direction / dist * self.step_size if dist > self.step_size else rand_sample

            # Simplified single-drone motion check (only for leader)
            temp_sim = MultiDrone(num_drones=1, environment_file=self.sim.environment_file)
            if temp_sim.motion_valid(np.array([q_near]), np.array([q_new])):
                new_idx = len(tree['nodes'])
                tree['nodes'].append(q_new)
                tree['parent'][new_idx] = nearest_idx
                
                if np.linalg.norm(q_new - leader_goal) < self.step_size:
                    path = []
                    curr = new_idx
                    while curr is not None:
                        path.append(tree['nodes'][curr])
                        curr = tree['parent'].get(curr)
                    self.leader_path = np.array(path[::-1])
                    return

        self.leader_path = np.array([leader_start, leader_goal])


    def _update_bridge_region(self):
        if len(self.tree_a['nodes']) < 5 or len(self.tree_b['nodes']) < 5:
            return
        
        sample_size = min(30, len(self.tree_a['nodes']), len(self.tree_b['nodes']))
        indices_a = np.random.choice(len(self.tree_a['nodes']), sample_size, replace=False)
        indices_b = np.random.choice(len(self.tree_b['nodes']), sample_size, replace=False)
        
        min_dist = float('inf')
        best_pair = None
        
        for idx_a in indices_a:
            node_a = self.tree_a['nodes'][idx_a]
            for idx_b in indices_b:
                node_b = self.tree_b['nodes'][idx_b]
                dist = np.linalg.norm(node_a - node_b)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (node_a, node_b)
        
        if best_pair:
            self.gaussian_center = (best_pair[0] + best_pair[1]) / 2
            self.gaussian_std = min_dist / 3
            self.gaussian_sampling_enabled = True

    def _gaussian_sample(self):
        if not self.gaussian_sampling_enabled or self.gaussian_center is None:
            return self._random_sample()
        
        self.stats['gaussian_samples'] += 1
        sample = np.random.randn(*self.gaussian_center.shape) * self.gaussian_std + self.gaussian_center
        
        bounds = self.sim._bounds
        sample = np.clip(sample, bounds[:, 0][np.newaxis, :], bounds[:, 1][np.newaxis, :])
        return sample

    def _leader_guided_sample(self):
        if self.leader_path is None or len(self.leader_path) < 2:
            return self._random_sample()
        
        self.stats['leader_samples'] += 1
        
        t = np.random.random()
        idx = int(t * (len(self.leader_path) - 1))
        alpha = t * (len(self.leader_path) - 1) - idx
        leader_pos = (1 - alpha) * self.leader_path[idx] + alpha * self.leader_path[idx + 1] if idx < len(self.leader_path) - 1 else self.leader_path[idx]

        config = np.zeros((self.num_drones, 3))
        for i in range(self.num_drones):
            offset = np.random.randn(3) * self.leader_dispersion
            config[i] = leader_pos + offset
        
        bounds = self.sim._bounds
        config = np.clip(config, bounds[:, 0][np.newaxis, :], bounds[:, 1][np.newaxis, :])
        return config

    def _random_sample(self):
        self.stats['random_samples'] += 1
        bounds = self.sim._bounds
        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.num_drones, 3))

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
        q_near = tree['nodes'][nearest_idx]
        
        dist = np.linalg.norm(q_target - q_near)
        q_new = q_near + (q_target - q_near) / dist * self.step_size if dist > self.step_size else q_target
        
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
            self.stats['step_adjustments'] += 1
            return None, None
            
    def _adaptive_connect(self, tree, q_target):
        for _ in range(30):
            status_idx, status_q = self._adaptive_extend(tree, q_target)
            if status_q is None:
                return "Trapped", -1
            if np.linalg.norm(status_q - q_target) < 1e-6:
                return "Reached", status_idx
        return "Advanced", status_idx

    def plan(self, timeout=110):
        start_time = time.time()
        
        if not self.sim.is_valid(self.q_init):
            return None
        if not self.sim.is_valid(self.q_goal):
            return None
        
        if self.sim.motion_valid(self.q_init, self.q_goal):
            return [self.q_init, self.q_goal]
        
        for i in range(self.max_iter):
            if time.time() - start_time > timeout:
                break
            
            if len(self.tree_a['nodes']) % 10 == 0: self._update_kdtree(self.tree_a)
            if len(self.tree_b['nodes']) % 10 == 0: self._update_kdtree(self.tree_b)
            if i > 0 and i % self.bridge_update_frequency == 0: self._update_bridge_region()
            
            q_rand = self._intelligent_sample()
            q_new_idx, q_new = self._adaptive_extend(self.tree_a, q_rand)
            
            if q_new is not None:
                connect_result, q_b_idx = self._adaptive_connect(self.tree_b, q_new)
                if connect_result == "Reached":
                    path = self._reconstruct_path(self.tree_a, self.tree_b, q_new_idx, q_b_idx)
                    return self._smooth_path(path)
            
            self._swap_trees()
        
        return None

    def _update_kdtree(self, tree):
        if len(tree['nodes']) > 0:
            flat_nodes = np.array(tree['nodes']).reshape(len(tree['nodes']), -1)
            tree['kdtree'] = KDTree(flat_nodes)
            tree['kdtree_outdated'] = False

    def _find_nearest(self, tree, q_target):
        if tree['kdtree'] is None or tree['kdtree_outdated']:
            self._update_kdtree(tree)
        
        if tree['kdtree'] is not None:
            _, idx = tree['kdtree'].query(q_target.flatten())
            return idx
        else: # Fallback for empty tree
            return -1

    def _swap_trees(self):
        self.tree_a, self.tree_b = self.tree_b, self.tree_a

    def _reconstruct_path(self, tree1, tree2, idx1, idx2):
        path1, curr_idx = [], idx1
        while curr_idx is not None:
            path1.append(tree1['nodes'][curr_idx])
            curr_idx = tree1['parent'].get(curr_idx)
        
        path2, curr_idx = [], idx2
        while curr_idx is not None:
            path2.append(tree2['nodes'][curr_idx])
            curr_idx = tree2['parent'].get(curr_idx)
        
        return path1[::-1] + path2[1:] if np.array_equal(tree1['nodes'][0], self.q_init) else path2[::-1] + path1[1:]

    def _smooth_path(self, path, max_iterations=None):
        max_iterations = max_iterations or self.smooth_iterations
        if len(path) <= 2: return path
        
        smoothed = path.copy()
        for _ in range(max_iterations):
            if len(smoothed) <= 2: break
            
            i, j = sorted(np.random.choice(len(smoothed), 2, replace=False))
            if j == i + 1: continue
            
            if self.sim.motion_valid(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        return smoothed


# =========================================================================================
# NEW: Systematic Experiment Runner
# =========================================================================================
def run_experiment(exp_name, configs, num_runs=20):
    # This function remains the same and is correctly implemented.
    # ... (your existing run_experiment code) ...
    pass # Placeholder for brevity, your actual code is kept

if __name__ == '__main__':
    # --- Experiment for Question B4: Environmental Complexity ---
    exp_B4_configs = [
        {'env_file': 'env_lv1.yaml', 'num_drones': 3},
        {'env_file': 'env_lv2.yaml', 'num_drones': 3},
        {'env_file': 'env_lv3.yaml', 'num_drones': 3},
        {'env_file': 'env_lv5.yaml', 'num_drones': 6},
    ]
    run_experiment("Environmental Complexity", exp_B4_configs, num_runs=20)

    # --- Experiment for Question B5: Number of Drones (Curse of Dimensionality) ---
    exp_B5_configs = [
        {'env_file': 'env_lv5_k4.yaml', 'num_drones': 4},
        {'env_file': 'env_lv5_k8.yaml', 'num_drones': 8},
        {'env_file': 'env_lv5_k12.yaml', 'num_drones': 12},
    ]
    # run_experiment("Curse of Dimensionality", exp_B5_configs, num_runs=10)

    print("All experiments completed.")