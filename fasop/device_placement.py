"""
Device placement and cluster configuration utilities.
"""

# Instance type to GPU type mapping for internal use
INSTANCE_TO_GPU = {
    "p4d.24xlarge": "A100",
    "g5.12xlarge": "A10",
    "g5.24xlarge": "A10",
    "A40": "A10",  # A40 uses A10 cost model internally
}


def get_gpu_for_stage(pp, N, node_type):
    """
    Determine GPU type for each pipeline stage based on node types.
    
    Args:
        pp: Pipeline parallelism degree
        N: Number of nodes
        node_type: List of instance types for each node
    
    Returns:
        List of GPU types ('A100' or 'A10') for each stage
    """
    if pp == 1:
        gpu = 'A100'
        for node_idx in range(N):
            if node_type[node_idx] in ['g5.12xlarge', 'g5.24xlarge', 'A40']:
                gpu = 'A10'
        return [gpu]
    
    gpu_for_stage = []
    for stage in range(pp):
        if pp < N:
            stage_per_node = N / pp
            gpu = 'A100'
            for node_idx in range(int(stage_per_node * stage), int(stage_per_node * (stage + 1))):
                if node_type[node_idx] in ['g5.12xlarge', 'g5.24xlarge', 'A40']:
                    gpu = 'A10'
            gpu_for_stage.append(gpu)
        elif pp > N:
            node_per_pp = pp / N
            node_idx = int(stage // node_per_pp)
            if node_type[node_idx] == 'p4d.24xlarge':
                gpu_for_stage.append('A100')
            else:
                gpu_for_stage.append('A10')
        else:
            node_idx = stage
            if node_type[node_idx] == 'p4d.24xlarge':
                gpu_for_stage.append('A100')
            else:
                gpu_for_stage.append('A10')
    
    return gpu_for_stage


def get_all_cluster_combinations(gpu_cluster: dict, pareto: bool = False):
    """
    Returns all possible cluster combinations for the given GPU cluster specification.
    
    Args:
        gpu_cluster: dict like {"A40": 8, "A100": 1} specifying GPU types and counts (REQUIRED)
        pareto: If True, enumerate all combinations up to the specified counts
    
    Returns:
        List of gpu_cluster dicts, e.g., [{"A40": 8}, {"A40": 7, "A100": 1}, ...]
    
    Raises:
        ValueError: If gpu_cluster is empty or None
    
    Example:
        # Pareto mode with {"A40": 2, "A100": 1}
        -> [{"A40": 1}, {"A40": 2}, {"A100": 1}, {"A40": 1, "A100": 1}, {"A40": 2, "A100": 1}]
        
        # Non-pareto mode
        -> [{"A40": 2, "A100": 1}]  # Returns input as-is
    """
    if not gpu_cluster:
        raise ValueError("gpu_cluster must be provided and non-empty. Use --gpus argument.")
    
    total_gpus = sum(gpu_cluster.values())
    if total_gpus == 0:
        raise ValueError("gpu_cluster must have at least one GPU.")
    
    # Non-pareto mode: return input cluster as-is
    if not pareto:
        return [gpu_cluster.copy()]
    
    # Pareto mode: enumerate all combinations up to the max counts
    # Get GPU types and their max counts
    gpu_types = list(gpu_cluster.keys())
    max_counts = [gpu_cluster[gpu_type] for gpu_type in gpu_types]
    
    cluster_combinations = []
    
    if len(gpu_types) == 1:
        # Single GPU type: enumerate 1 to max
        gpu_type = gpu_types[0]
        for count in range(1, max_counts[0] + 1):
            cluster_combinations.append({gpu_type: count})
    
    elif len(gpu_types) == 2:
        # Two GPU types: enumerate all combinations
        for count0 in range(0, max_counts[0] + 1):
            for count1 in range(0, max_counts[1] + 1):
                if count0 + count1 == 0:
                    continue
                combo = {}
                if count0 > 0:
                    combo[gpu_types[0]] = count0
                if count1 > 0:
                    combo[gpu_types[1]] = count1
                cluster_combinations.append(combo)
    
    else:
        # More than 2 GPU types: use recursive approach
        def enumerate_combinations(type_idx, current_combo):
            if type_idx == len(gpu_types):
                if sum(current_combo.values()) > 0:
                    cluster_combinations.append(current_combo.copy())
                return
            
            gpu_type = gpu_types[type_idx]
            for count in range(0, max_counts[type_idx] + 1):
                if count > 0:
                    current_combo[gpu_type] = count
                enumerate_combinations(type_idx + 1, current_combo)
                if count > 0:
                    del current_combo[gpu_type]
        
        enumerate_combinations(0, {})
    
    print(f"Number of cluster combinations (pareto): {len(cluster_combinations)}")
    return cluster_combinations


def device_placement_from_cluster(gpu_cluster: dict):
    """
    Generate device placement permutations from gpu_cluster specification.
    
    Args:
        gpu_cluster: dict like {"A40": 8, "A100": 1} specifying GPU types and counts
    
    Returns:
        List of device placement lists, where each element is a GPU type string
        
    Example:
        gpu_cluster = {"A40": 2, "A100": 1}
        -> [['A100', 'A40', 'A40'], ['A40', 'A100', 'A40'], ['A40', 'A40', 'A100']]
    """
    # Build node list from gpu_cluster
    nodes = []
    gpu_types = sorted(gpu_cluster.keys())  # Consistent ordering
    
    for gpu_type in gpu_types:
        count = gpu_cluster[gpu_type]
        nodes.extend([gpu_type] * count)
    
    print(f"GPU nodes: {nodes}")
    
    # If only one GPU type, no permutations needed
    if len(gpu_types) == 1:
        return [nodes]
    
    # Generate cyclic permutations for heterogeneous clusters
    return cyclic_permutation(nodes)


def device_placement(num_a100: int, num_other: int):
    """
    Legacy interface for device placement.
    Use device_placement_from_cluster() for new code.
    
    Args:
        num_a100: Number of A100 GPUs
        num_other: Number of other GPUs (A10, A40, etc.)
    
    Returns:
        List of device placement lists using 'A' (A100) and 'B' (other) markers
    """
    a100_nodes = ['A'] * num_a100
    other_nodes = ['B'] * num_other
    all_nodes = a100_nodes + other_nodes

    print(f"Device nodes (A=A100, B=other): {all_nodes}")

    if num_a100 * num_other == 0:
        return [all_nodes]
    else:
        return cyclic_permutation(all_nodes)


def device_placement_all_from_cluster(gpu_cluster: dict):
    """
    Generate ALL unique device placement permutations from gpu_cluster specification.
    
    Args:
        gpu_cluster: dict like {"A40": 8, "A100": 1} specifying GPU types and counts
    
    Returns:
        List of all unique device placement permutations
    """
    # Build node list from gpu_cluster
    nodes = []
    gpu_types = sorted(gpu_cluster.keys())
    
    for gpu_type in gpu_types:
        count = gpu_cluster[gpu_type]
        nodes.extend([gpu_type] * count)
    
    print(f"GPU nodes: {nodes}")
    
    # If only one GPU type, no permutations needed
    if len(gpu_types) == 1:
        return [nodes]
    
    # Generate all unique permutations
    D = []
    # Create a string representation for msp
    node_str = ''.join(['A' if g == gpu_types[0] else 'B' for g in nodes])
    
    for perm in msp(node_str):
        placement = []
        for val in perm:
            if val == 0:
                placement.append(gpu_types[0])
            else:
                placement.append(gpu_types[1] if len(gpu_types) > 1 else gpu_types[0])
        D.append(placement)
    
    return D


def device_placement_all(num_a100: int, num_other: int):
    """
    Legacy interface for generating all unique device placements.
    Use device_placement_all_from_cluster() for new code.
    """
    a100_nodes = 'A' * num_a100
    other_nodes = 'B' * num_other
    all_nodes = a100_nodes + other_nodes

    print(f"Device nodes (A=A100, B=other): {all_nodes}")

    D = []

    if num_a100 * num_other == 0:
        d = list(all_nodes)
        D.append(d)
        return D
    else:
        for d in msp(all_nodes):
            de = ['A' if i == 1 else 'B' for i in d]
            D.append(de)
    return D


def cyclic_permutation(l):
    """
    Returns all cyclic permutations of the given list
    """
    permutations = []
    count = 0
    for i in range(len(l)):
        permutations.append(l[i:] + l[:i])
        count += 1
    print(f"Number of node placement: {count}")
    return permutations


def msp(items):
  '''Yield the permutations of `items` where items is either a list
  of integers representing the actual items or a list of hashable items.
  The output are the unique permutations of the items given as a list
  of integers 0, ..., n-1 that represent the n unique elements in
  `items`.

  Examples
  ========

  >>> for i in msp('xoxox'):
  ...   print(i)

  [1, 1, 1, 0, 0]
  [0, 1, 1, 1, 0]
  [1, 0, 1, 1, 0]
  [1, 1, 0, 1, 0]
  [0, 1, 1, 0, 1]
  [1, 0, 1, 0, 1]
  [0, 1, 0, 1, 1]
  [0, 0, 1, 1, 1]
  [1, 0, 0, 1, 1]
  [1, 1, 0, 0, 1]

  Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
  https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
  '''

  def visit(head):
      (rv, j) = ([], head)
      for i in range(N):
          (dat, j) = E[j]
          rv.append(dat)
      return rv

  u = list(set(items))
  E = list(([u.index(i) for i in items]))
  N = len(E)
  # put E into linked-list format
  (val, nxt) = (0, 1)
  for i in range(N):
      E[i] = [E[i], i + 1]
  E[-1][nxt] = None
  head = 0
  afteri = N - 1
  i = afteri - 1
  yield visit(head)
  while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
      j = E[afteri][nxt]  # added to algorithm for clarity
      if j is not None and E[i][val] >= E[j][val]:
          beforek = afteri
      else:
          beforek = i
      k = E[beforek][nxt]
      E[beforek][nxt] = E[k][nxt]
      E[k][nxt] = head
      if E[k][val] < E[head][val]:
          i = k
      afteri = E[i][nxt]
      head = k
      yield visit(head)
