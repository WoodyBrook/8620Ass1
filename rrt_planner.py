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
        print(f"Computing leader path for {self.num_drones} drones...")
        
        # Create a simplified problem with just one drone (the center one)
        if self.num_drones > 0:
            # Use the middle drone as leader
            leader_idx = self.num_drones // 2
            leader_start = self.q_init[leader_idx:leader_idx+1]
            leader_goal = self.q_goal[leader_idx:leader_idx+1]
            
            # Simple RRT for single drone (fast)
            leader_tree = [leader_start]
            leader_parent = {0: None}
            
            for i in range(2000):  # Quick planning with limited iterations
                # Sample
                if np.random.random() < 0.1:
                    target = leader_goal
                else:
                    bounds = self.sim._bounds
                    target = np.random.uniform(bounds[:, 0], bounds[:, 1])
                
                # Find nearest
                distances = [np.linalg.norm(node - target) for node in leader_tree]
                nearest_idx = np.argmin(distances)
                nearest = leader_tree[nearest_idx]
                
                # Extend
                direction = target - nearest
                dist = np.linalg.norm(direction)
                if dist > self.step_size:
                    new_pos = nearest + direction / dist * self.step_size
                else:
                    new_pos = target
                
                # Simple collision check (just bounds for leader path)
                if np.all(new_pos >= self.sim._bounds[:, 0]) and np.all(new_pos <= self.sim._bounds[:, 1]):
                    leader_tree.append(new_pos)
                    leader_parent[len(leader_tree)-1] = nearest_idx
                    
                    # Check if reached goal
                    if np.linalg.norm(new_pos - leader_goal) < self.step_size:
                        # Reconstruct path
                        path = []
                        curr = len(leader_tree) - 1
                        while curr is not None:
                            path.append(leader_tree[curr])
                            curr = leader_parent.get(curr)
                        self.leader_path = path[::-1]
                        print(f"Leader path found with {len(self.leader_path)} waypoints")
                        return
            
            print("Leader path not found, using direct line")
            self.leader_path = [leader_start, leader_goal]

    def _update_bridge_region(self):
        """
        Strategy 1: Find the closest pair of nodes between trees for Gaussian sampling.
        """
        if len(self.tree_a['nodes']) < 5 or len(self.tree_b['nodes']) < 5:
            return
        
        # Sample nodes from each tree for efficiency
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
            # Set Gaussian sampling center to midpoint of closest nodes
            self.gaussian_center = (best_pair[0] + best_pair[1]) / 2
            self.gaussian_std = min_dist / 3  # Adjust std based on gap size
            self.gaussian_sampling_enabled = True
            print(f"Bridge region updated: gap={min_dist:.2f}, std={self.gaussian_std:.2f}")

    def _gaussian_sample(self):
        """
        Strategy 1: Sample from Gaussian distribution around bridge region.
        """
        if not self.gaussian_sampling_enabled or self.gaussian_center is None:
            return self._random_sample()
        
        self.stats['gaussian_samples'] += 1
        
        # Sample from Gaussian distribution centered at bridge
        sample = np.random.randn(*self.gaussian_center.shape) * self.gaussian_std + self.gaussian_center
        
        # Clip to bounds
        bounds = self.sim._bounds
        for i in range(self.num_drones):
            sample[i] = np.clip(sample[i], bounds[:, 0], bounds[:, 1])
        
        return sample

    def _leader_guided_sample(self):
        """
        Strategy 2: Sample configurations guided by the leader path.
        """
        if self.leader_path is None or len(self.leader_path) < 2:
            return self._random_sample()
        
        self.stats['leader_samples'] += 1
        
        # Pick a random point along the leader path
        t = np.random.random()
        idx = int(t * (len(self.leader_path) - 1))
        if idx < len(self.leader_path) - 1:
            # Interpolate between waypoints
            alpha = t * (len(self.leader_path) - 1) - idx
            leader_pos = (1 - alpha) * self.leader_path[idx] + alpha * self.leader_path[idx + 1]
        else:
            leader_pos = self.leader_path[idx]
        
        # Create configuration for all drones around the leader position
        config = np.zeros((self.num_drones, 3))
        for i in range(self.num_drones):
            # Add dispersion around leader position
            offset = np.random.randn(3) * self.leader_dispersion
            config[i] = leader_pos.flatten() + offset
        
        # Clip to bounds
        bounds = self.sim._bounds
        for i in range(self.num_drones):
            config[i] = np.clip(config[i], bounds[:, 0], bounds[:, 1])
        
        return config

    def _random_sample(self):
        """Standard random sampling."""
        self.stats['random_samples'] += 1
        config = np.zeros((self.num_drones, 3))
        bounds = self.sim._bounds
        for i in range(self.num_drones):
            config[i, 0] = np.random.uniform(bounds[0, 0], bounds[0, 1])
            config[i, 1] = np.random.uniform(bounds[1, 0], bounds[1, 1])
            config[i, 2] = np.random.uniform(bounds[2, 0], bounds[2, 1])
        return config

    def _intelligent_sample(self):
        """
        Combine all sampling strategies with appropriate probabilities.
        """
        rand = np.random.random()
        
        if rand < self.goal_bias:
            # Goal biasing
            self.stats['goal_samples'] += 1
            return self.q_goal
        elif rand < self.goal_bias + self.leader_sample_ratio and self.leader_path is not None:
            # Leader-guided sampling
            return self._leader_guided_sample()
        elif rand < self.goal_bias + self.leader_sample_ratio + self.gaussian_sample_ratio and self.gaussian_sampling_enabled:
            # Gaussian bridge sampling
            return self._gaussian_sample()
        else:
            # Random sampling
            return self._random_sample()

    def _adaptive_extend(self, tree, q_target):
        """
        Strategy 3: Extend with adaptive step size.
        Increase step size on success, decrease on failure.
        """
        nearest_idx = self._find_nearest(tree, q_target)
        q_near = tree['nodes'][nearest_idx]
        
        # Steer with current step size
        dist = np.linalg.norm(q_target - q_near)
        if dist < self.step_size:
            q_new = q_target
        else:
            q_new = q_near + (q_target - q_near) / dist * self.step_size
        
        # Try to extend
        if self.sim.motion_valid(q_near, q_new):
            # Success: increase step size slightly
            self.step_size = min(self.step_size * self.step_increase_rate, self.max_step)
            self.stats['successful_extends'] += 1
            
            # Add to tree
            new_idx = len(tree['nodes'])
            tree['nodes'].append(q_new)
            tree['parent'][new_idx] = nearest_idx
            tree['kdtree_outdated'] = True
            
            return new_idx, q_new
        else:
            # Failure: decrease step size
            self.step_size = max(self.step_size * self.step_decrease_rate, self.min_step)
            self.stats['failed_extends'] += 1
            self.stats['step_adjustments'] += 1
            
            # Try again with smaller step
            if dist > self.step_size:
                q_new_retry = q_near + (q_target - q_near) / dist * self.step_size
                if self.sim.motion_valid(q_near, q_new_retry):
                    new_idx = len(tree['nodes'])
                    tree['nodes'].append(q_new_retry)
                    tree['parent'][new_idx] = nearest_idx
                    tree['kdtree_outdated'] = True
                    return new_idx, q_new_retry
            
            return None, None

    def plan(self, timeout=110):
        """
        Main planning loop with all three strategies integrated.
        """
        start_time = time.time()
        
        # Validation
        if not self.sim.is_valid(self.q_init):
            print("Initial configuration is invalid!")
            return None
        if not self.sim.is_valid(self.q_goal):
            print("Goal configuration is invalid!")
            return None
        
        # Try direct connection first
        if self.sim.motion_valid(self.q_init, self.q_goal):
            print("Direct path found!")
            return [self.q_init, self.q_goal]
        
        print(f"Starting advanced planning with {self.num_drones} drones...")
        print(f"Strategies: Gaussian={self.gaussian_sample_ratio}, Leader={self.leader_sample_ratio}, Goal={self.goal_bias}")
        
        for i in range(self.max_iter):
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"Timeout after {i} iterations")
                break
            
            # Update KD-Trees periodically
            if len(self.tree_a['nodes']) % 10 == 0:
                self._update_kdtree(self.tree_a)
            if len(self.tree_b['nodes']) % 10 == 0:
                self._update_kdtree(self.tree_b)
            
            # Update bridge region for Gaussian sampling
            if i > 0 and i % self.bridge_update_frequency == 0:
                self._update_bridge_region()
            
            # Intelligent sampling combining all strategies
            q_rand = self._intelligent_sample()
            
            # Extend with adaptive step size
            q_new_idx, q_new = self._adaptive_extend(self.tree_a, q_rand)
            
            if q_new is not None:
                # Try to connect
                connect_result, q_b_idx = self._adaptive_connect(self.tree_b, q_new)
                if connect_result == "Reached":
                    elapsed = time.time() - start_time
                    print(f"Path found in {i+1} iterations, {elapsed:.2f}s")
                    print(f"Final step size: {self.step_size:.3f}")
                    print(f"Sampling stats: {self.stats}")
                    
                    path = self._reconstruct_path(self.tree_a, self.tree_b, q_new_idx, q_b_idx)
                    smoothed = self._smooth_path(path)
                    print(f"Path: {len(path)} → {len(smoothed)} waypoints")
                    
                    return smoothed
            
            # Swap trees
            self._swap_trees()
            
            # Progress report
            if i > 0 and i % 1000 == 0:
                print(f"Iteration {i}: Trees have {len(self.tree_a['nodes'])} and {len(self.tree_b['nodes'])} nodes")
                print(f"Current step size: {self.step_size:.3f}")
        
        print(f"Failed to find path. Final stats: {self.stats}")
        return None

    def _adaptive_connect(self, tree, q_target):
        """Connect with adaptive step size."""
        max_extensions = 30
        
        for _ in range(max_extensions):
            status_idx, status_q = self._adaptive_extend(tree, q_target)
            if status_q is None:
                return "Trapped", -1
            
            if np.linalg.norm(status_q - q_target) < 1e-6:
                return "Reached", status_idx
        
        return "Advanced", status_idx

    def _update_kdtree(self, tree):
        """Update KD-Tree for efficient nearest neighbor search."""
        if len(tree['nodes']) > 0:
            flat_nodes = np.array(tree['nodes']).reshape(len(tree['nodes']), -1)
            tree['kdtree'] = KDTree(flat_nodes)
            tree['kdtree_outdated'] = False

    def _find_nearest(self, tree, q_target):
        """Find nearest node using KD-Tree."""
        if tree['kdtree'] is None or tree['kdtree_outdated']:
            self._update_kdtree(tree)
        
        if tree['kdtree'] is not None and len(tree['nodes']) > 1:
            flat_target = q_target.flatten()
            _, idx = tree['kdtree'].query(flat_target)
            return idx
        else:
            nodes = np.array(tree['nodes'])
            dists = np.linalg.norm(nodes - q_target, axis=(1, 2) if nodes.ndim == 3 else 1)
            return np.argmin(dists)

    def _swap_trees(self):
        """Swap trees for bidirectional growth."""
        self.tree_a, self.tree_b = self.tree_b, self.tree_a

    def _reconstruct_path(self, tree1, tree2, idx1, idx2):
        """Reconstruct path from connected trees."""
        path1 = []
        curr_idx = idx1
        while curr_idx is not None:
            path1.append(tree1['nodes'][curr_idx])
            curr_idx = tree1['parent'].get(curr_idx)
        
        path2 = []
        curr_idx = idx2
        while curr_idx is not None:
            path2.append(tree2['nodes'][curr_idx])
            curr_idx = tree2['parent'].get(curr_idx)
        
        if np.array_equal(tree1['nodes'][0], self.q_init):
            return path1[::-1] + path2[1:]
        else:
            return path2[::-1] + path1[1:]

    def _smooth_path(self, path, max_iterations=None):
        """Path smoothing with shortcuts."""
        if max_iterations is None:
            max_iterations = self.smooth_iterations
        
        if len(path) <= 2:
            return path
        
        smoothed = path.copy()
        
        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break
            
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            
            if self.sim.motion_valid(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        return smoothed


if __name__ == '__main__':
    # Test with different drone counts
    for n_drones in [3, 6, 9, 12]:
        print(f"\n{'='*60}")
        print(f"Testing with {n_drones} drones")
        print(f"{'='*60}")
        
        sim = MultiDrone(num_drones=n_drones, environment_file="env_lv2.yaml")
        
        planner = rrt_planner(
            sim,
            max_iter=30000,
            base_step_size=2.0,
            smooth_iterations=100
        )
        
        start = time.time()
        path = planner.plan(timeout=110)
        elapsed = time.time() - start
        
        if path:
            print(f"✅ Success in {elapsed:.2f}s")
        else:
            print(f"❌ Failed after {elapsed:.2f}s")