import numpy as np
import time
from scipy.spatial import KDTree
from multi_drone import MultiDrone

class RRTConnectPlanner:
    def __init__(self, sim, max_iter=20000, step_size=1.5, smooth_iterations=50):
        """
        RRT-Connect Planner for Multi-Drone Motion Planning with KD-Tree optimization.

        Args:
            sim (MultiDrone): An instance of the MultiDrone simulator.
            max_iter (int): The maximum number of iterations.
            step_size (float): The maximum step size for tree extension.
            smooth_iterations (int): Number of iterations for path smoothing.
        """
        self.sim = sim
        self.max_iter = max_iter
        self.step_size = step_size
        self.smooth_iterations = smooth_iterations
        
        self.q_init = self.sim.initial_configuration
        self.q_goal = self.sim.goal_positions  # Note: We use goal positions as the target config
        
        # Tree A starts from q_init, Tree B starts from q_goal
        self.tree_a = {
            'nodes': [self.q_init], 
            'parent': {0: None},
            'kdtree': None,
            'kdtree_outdated': True  # To track if KD-Tree needs update
        }
        self.tree_b = {
            'nodes': [self.q_goal], 
            'parent': {0: None},
            'kdtree': None,
            'kdtree_outdated': True
        }
        
        # KD-Tree update frequency (update every N nodes added)
        self.kdtree_update_frequency = 10

    def plan(self):
        """
        Executes the RRT-Connect planning algorithm with KD-Tree optimization.

        Returns:
            list[np.ndarray] or None: A list of configurations representing the path, 
                                      or None if no path is found.
        """
        start_time = time.time()
        
        # Check if initial and goal configurations are valid first
        if not self.sim.is_valid(self.q_init):
            print("Initial configuration is invalid!")
            return None
        if not self.sim.is_valid(self.q_goal):
            print("Goal configuration is invalid!")
            return None

        for i in range(self.max_iter):
            # Update KD-Trees periodically
            if len(self.tree_a['nodes']) % self.kdtree_update_frequency == 0:
                self._update_kdtree(self.tree_a)
            if len(self.tree_b['nodes']) % self.kdtree_update_frequency == 0:
                self._update_kdtree(self.tree_b)
            
            # 1. Sample a random configuration
            q_rand = self._sample_random_config()
            
            # 2. Extend Tree A and try to connect with Tree B
            q_new_idx, q_new = self._extend(self.tree_a, q_rand)
            
            if q_new is not None:  # If extension was successful
                connect_result, q_b_idx = self._connect(self.tree_b, q_new)
                if connect_result == "Reached":
                    elapsed_time = time.time() - start_time
                    print(f"Path found in {i+1} iterations and {elapsed_time:.2f} seconds!")
                    
                    # Reconstruct the path
                    path = self._reconstruct_path(self.tree_a, self.tree_b, q_new_idx, q_b_idx)
                    
                    # Smooth the path
                    print(f"Original path has {len(path)} waypoints.")
                    smoothed_path = self._smooth_path(path)
                    print(f"Smoothed path has {len(smoothed_path)} waypoints (reduced by {len(path) - len(smoothed_path)}).")
                    
                    return smoothed_path

            # 3. Swap the trees to grow the other one in the next iteration
            self._swap_trees()

        print(f"Failed to find a path after {self.max_iter} iterations.")
        return None

    def _update_kdtree(self, tree):
        """Updates the KD-Tree for efficient nearest neighbor search."""
        if len(tree['nodes']) > 0:
            # Flatten configurations for KD-Tree (convert from [N, K, 3] to [N, K*3])
            flat_nodes = np.array(tree['nodes']).reshape(len(tree['nodes']), -1)
            tree['kdtree'] = KDTree(flat_nodes)
            tree['kdtree_outdated'] = False

    def _sample_random_config(self):
        """Samples a random configuration within the workspace bounds."""
        config = np.zeros((self.sim.N, 3))
        bounds = self.sim._bounds
        for i in range(self.sim.N):
            config[i, 0] = np.random.uniform(bounds[0, 0], bounds[0, 1])
            config[i, 1] = np.random.uniform(bounds[1, 0], bounds[1, 1])
            config[i, 2] = np.random.uniform(bounds[2, 0], bounds[2, 1])
        return config

    def _find_nearest(self, tree, q_target):
        """
        Finds the index of the nearest node in the tree to the target configuration.
        Uses KD-Tree for efficient search in high-dimensional spaces.
        """
        # Update KD-Tree if needed
        if tree['kdtree'] is None or tree['kdtree_outdated']:
            self._update_kdtree(tree)
        
        # Use KD-Tree for fast nearest neighbor search
        if tree['kdtree'] is not None and len(tree['nodes']) > 1:
            flat_target = q_target.flatten()
            _, idx = tree['kdtree'].query(flat_target)
            return idx
        else:
            # Fallback to linear search for very small trees
            nodes = np.array(tree['nodes'])
            dists = np.linalg.norm(nodes - q_target, axis=(1, 2) if nodes.ndim == 3 else 1)
            return np.argmin(dists)

    def _steer(self, q_from, q_to):
        """Steers from q_from towards q_to by at most step_size."""
        dist = np.linalg.norm(q_to - q_from)
        if dist < self.step_size:
            return q_to
        else:
            return q_from + (q_to - q_from) / dist * self.step_size

    def _extend(self, tree, q_target):
        """Extends the tree by one step towards the target configuration."""
        nearest_idx = self._find_nearest(tree, q_target)
        q_near = tree['nodes'][nearest_idx]
        q_new = self._steer(q_near, q_target)
        
        if self.sim.motion_valid(q_near, q_new):
            new_idx = len(tree['nodes'])
            tree['nodes'].append(q_new)
            tree['parent'][new_idx] = nearest_idx
            tree['kdtree_outdated'] = True  # Mark KD-Tree as needing update
            return new_idx, q_new
        return None, None

    def _connect(self, tree, q_target):
        """Greedily tries to connect the tree to the target configuration."""
        while True:
            status_idx, status_q = self._extend(tree, q_target)
            if status_q is None:  # Trapped
                return "Trapped", -1
            
            dist_to_target = np.linalg.norm(status_q - q_target)
            if dist_to_target < 1e-6:  # Reached
                return "Reached", status_idx
            # Continue extending in the next loop

    def _swap_trees(self):
        """Swaps tree_a and tree_b."""
        self.tree_a, self.tree_b = self.tree_b, self.tree_a

    def _reconstruct_path(self, tree1, tree2, idx1, idx2):
        """Reconstructs the full path after the trees have connected."""
        # Trace back path from tree1
        path1 = []
        curr_idx = idx1
        while curr_idx is not None:
            path1.append(tree1['nodes'][curr_idx])
            curr_idx = tree1['parent'][curr_idx]
        
        # Trace back path from tree2
        path2 = []
        curr_idx = idx2
        while curr_idx is not None:
            path2.append(tree2['nodes'][curr_idx])
            curr_idx = tree2['parent'][curr_idx]

        # Check which tree was which and combine paths
        if np.array_equal(tree1['nodes'][0], self.q_init):  # tree1 is Ta
            return path1[::-1] + path2[1:]
        else:  # tree1 is Tb
            return path2[::-1] + path1[1:]

    def _smooth_path(self, path, max_iterations=None):
        """
        Smooths the path by attempting to remove intermediate waypoints.
        Uses shortcutting: tries to connect non-adjacent waypoints directly.
        
        Args:
            path: Original path as a list of configurations
            max_iterations: Number of smoothing iterations (uses self.smooth_iterations if None)
            
        Returns:
            Smoothed path with fewer waypoints
        """
        if max_iterations is None:
            max_iterations = self.smooth_iterations
            
        if len(path) <= 2:
            return path
        
        smoothed = path.copy()
        
        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break
                
            # Randomly select two non-adjacent waypoints
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            
            # Try to connect them directly
            if self.sim.motion_valid(smoothed[i], smoothed[j]):
                # Remove intermediate waypoints
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        # Additional pass: try to shortcut from start to as far as possible
        final_smoothed = [smoothed[0]]
        i = 0
        while i < len(smoothed) - 1:
            # Find the farthest point we can reach directly
            for j in range(len(smoothed) - 1, i, -1):
                if self.sim.motion_valid(smoothed[i], smoothed[j]):
                    final_smoothed.append(smoothed[j])
                    i = j
                    break
            else:
                # If no shortcut found, take the next point
                i += 1
                if i < len(smoothed):
                    final_smoothed.append(smoothed[i])
        
        # Ensure the path ends exactly at the goal
        if not np.array_equal(final_smoothed[-1], path[-1]):
            final_smoothed.append(path[-1])
        
        return final_smoothed


if __name__ == '__main__':
    # Initialize the MultiDrone environment from the YAML file
    # You can change the environment_file for your experiments
    sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

    # Create the planner with optimizations
    planner = RRTConnectPlanner(
        sim, 
        max_iter=20000, 
        step_size=2.0,
        smooth_iterations=100  # Increase for more aggressive smoothing
    )
    
    # Plan a path
    print("Starting RRT-Connect planning with KD-Tree optimization...")
    path = planner.plan()
    
    # Visualize the path if found
    if path:
        print(f"Final path has {len(path)} waypoints.")
        # Ensure the path starts exactly at the initial configuration
        path[0] = sim.initial_configuration
        # Ensure the path ends exactly at the goal
        path[-1] = sim.goal_positions
        sim.visualize_paths(path)
    else:
        print("Could not find a path.")