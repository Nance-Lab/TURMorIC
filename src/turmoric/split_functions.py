"""
Common train/test split functions for VAMPIRE cross-validation.

This module provides ready-to-use split functions and examples for custom splits.
All split functions have the signature:
    split_function(group_ids: List[str], k: int, random_seed: Optional[int]) 
        -> List[Tuple[List[str], List[str]]]
"""

from typing import List, Tuple, Optional
import random


# =============================================================================
# BUILT-IN SPLIT FUNCTIONS
# =============================================================================

def kfold_split(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None
) -> List[Tuple[List[str], List[str]]]:
    """
    Standard k-fold cross-validation split.
    
    Divides group IDs into k roughly equal folds. Each fold serves as
    the test set once, with remaining folds as training.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers (slices, animals, etc.)
    k : int
        Number of folds
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    splits : list of tuple
        List of (train_groups, test_groups) tuples, one per fold
        
    Examples
    --------
    >>> groups = ['slice1', 'slice2', 'slice3', 'slice4', 'slice5']
    >>> splits = kfold_split(groups, k=5, random_seed=42)
    >>> len(splits)
    5
    >>> # Each split has 4 training groups and 1 test group
    """
    if len(group_ids) < k:
        raise ValueError(
            f"Cannot split {len(group_ids)} groups into {k} folds"
        )
    
    # Shuffle group IDs
    shuffled = group_ids.copy()
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(shuffled)
    
    # Split into k folds
    fold_size = len(shuffled) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(shuffled)
        folds.append(shuffled[start:end])
    
    # Create train/test splits
    splits = []
    for i in range(k):
        test = folds[i]
        train = [g for j, fold in enumerate(folds) if j != i for g in fold]
        splits.append((train, test))
    
    return splits


def leave_one_out_split(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None
) -> List[Tuple[List[str], List[str]]]:
    """
    Leave-one-out cross-validation (LOOCV).
    
    Each group serves as the test set once. Equivalent to k-fold where k = n_groups.
    The k parameter is ignored.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Ignored (kept for signature compatibility)
    random_seed : int, optional
        Random seed for order (optional, doesn't affect results)
        
    Returns
    -------
    splits : list of tuple
        List of (train_groups, test_groups) tuples, one per group
        
    Examples
    --------
    >>> groups = ['slice1', 'slice2', 'slice3']
    >>> splits = leave_one_out_split(groups, k=None, random_seed=42)
    >>> len(splits)
    3
    >>> # Each split has n-1 training groups and 1 test group
    """
    splits = []
    
    # Optionally shuffle order
    ordered_groups = group_ids.copy()
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(ordered_groups)
    
    for test_group in ordered_groups:
        train = [g for g in ordered_groups if g != test_group]
        test = [test_group]
        splits.append((train, test))
    
    return splits


def stratified_split_by_size(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None,
    group_sizes: Optional[dict] = None
) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified k-fold split that balances fold sizes by group sample count.
    
    Useful when groups have very different numbers of images. Ensures each
    fold has roughly equal total sample counts (not just equal number of groups).
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Number of folds
    random_seed : int, optional
        Random seed for reproducibility
    group_sizes : dict, optional
        Dictionary mapping group_id -> number of samples.
        If None, treats all groups as equal size.
        
    Returns
    -------
    splits : list of tuple
        List of (train_groups, test_groups) tuples
        
    Examples
    --------
    >>> groups = ['slice1', 'slice2', 'slice3', 'slice4']
    >>> sizes = {'slice1': 100, 'slice2': 50, 'slice3': 150, 'slice4': 75}
    >>> splits = stratified_split_by_size(groups, k=2, group_sizes=sizes)
    >>> # Folds will have similar total sample counts
    """
    if group_sizes is None:
        # Fall back to regular k-fold
        return kfold_split(group_ids, k, random_seed)
    
    # Sort groups by size (descending)
    sorted_groups = sorted(group_ids, key=lambda g: group_sizes.get(g, 0), reverse=True)
    
    # Initialize fold sample counts
    fold_totals = [0] * k
    folds = [[] for _ in range(k)]
    
    # Greedy bin packing: assign each group to smallest fold
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(sorted_groups)  # Add randomness to sorted order
    
    for group in sorted_groups:
        # Find fold with smallest total
        min_fold_idx = fold_totals.index(min(fold_totals))
        folds[min_fold_idx].append(group)
        fold_totals[min_fold_idx] += group_sizes.get(group, 1)
    
    # Create train/test splits
    splits = []
    for i in range(k):
        test = folds[i]
        train = [g for j, fold in enumerate(folds) if j != i for g in fold]
        splits.append((train, test))
    
    return splits


def single_train_test_split(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None,
    test_size: float = 0.2
) -> List[Tuple[List[str], List[str]]]:
    """
    Simple single train/test split (not cross-validation).
    
    Returns a single split with specified test proportion.
    The k parameter is ignored.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Ignored (kept for signature compatibility)
    random_seed : int, optional
        Random seed for reproducibility
    test_size : float, optional
        Proportion of groups to use for testing (default: 0.2)
        
    Returns
    -------
    splits : list of tuple
        List containing one (train_groups, test_groups) tuple
        
    Examples
    --------
    >>> groups = ['slice1', 'slice2', 'slice3', 'slice4', 'slice5']
    >>> splits = single_train_test_split(groups, k=None, test_size=0.2)
    >>> len(splits)
    1
    >>> train, test = splits[0]
    >>> len(test) / len(groups)  # Approximately 0.2
    0.2
    """
    shuffled = group_ids.copy()
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(shuffled)
    
    n_test = max(1, int(len(shuffled) * test_size))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    
    return [(train, test)]


def leave_one_group_out(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None,
    group_to_leave_out: Optional[str] = None
) -> List[Tuple[List[str], List[str]]]:
    """
    Leave one specific group out for testing, train on all others.
    
    Useful for testing generalization to a specific experimental condition.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Ignored (kept for signature compatibility)
    random_seed : int, optional
        Ignored
    group_to_leave_out : str, optional
        Which group to use as test set. If None, uses first group.
        
    Returns
    -------
    splits : list of tuple
        List containing one (train_groups, test_groups) tuple
        
    Examples
    --------
    >>> groups = ['control_slice1', 'control_slice2', 'treated_slice1']
    >>> splits = leave_one_group_out(groups, k=None, group_to_leave_out='treated_slice1')
    >>> train, test = splits[0]
    >>> test
    ['treated_slice1']
    """
    if group_to_leave_out is None:
        group_to_leave_out = group_ids[0]
    
    if group_to_leave_out not in group_ids:
        raise ValueError(f"Group '{group_to_leave_out}' not in group_ids")
    
    test = [group_to_leave_out]
    train = [g for g in group_ids if g != group_to_leave_out]
    
    return [(train, test)]


def monte_carlo_split(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None,
    n_iterations: int = 10,
    test_size: float = 0.2
) -> List[Tuple[List[str], List[str]]]:
    """
    Monte Carlo cross-validation: repeated random train/test splits.
    
    Creates multiple random splits. The k parameter is replaced by n_iterations.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Ignored (use n_iterations instead)
    random_seed : int, optional
        Random seed for reproducibility
    n_iterations : int, optional
        Number of random splits to create (default: 10)
    test_size : float, optional
        Proportion of groups for testing (default: 0.2)
        
    Returns
    -------
    splits : list of tuple
        List of (train_groups, test_groups) tuples, one per iteration
        
    Examples
    --------
    >>> groups = ['slice1', 'slice2', 'slice3', 'slice4', 'slice5']
    >>> splits = monte_carlo_split(groups, k=None, n_iterations=3, test_size=0.2)
    >>> len(splits)
    3
    >>> # Each split has a random 80/20 train/test split
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    splits = []
    n_test = max(1, int(len(group_ids) * test_size))
    
    for i in range(n_iterations):
        shuffled = group_ids.copy()
        random.shuffle(shuffled)
        
        test = shuffled[:n_test]
        train = shuffled[n_test:]
        splits.append((train, test))
    
    return splits


# =============================================================================
# CUSTOM SPLIT FUNCTION TEMPLATE
# =============================================================================

def custom_split_template(
    group_ids: List[str], 
    k: int, 
    random_seed: Optional[int] = None
) -> List[Tuple[List[str], List[str]]]:
    """
    Template for creating custom split functions.
    
    Copy this function and modify the logic to create your custom splits.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    k : int
        Number of splits (or other parameter - use as needed)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    splits : list of tuple
        List of (train_groups, test_groups) tuples
        
    Notes
    -----
    Your function MUST:
    1. Take exactly these three parameters
    2. Return a list of (train_list, test_list) tuples
    3. Ensure every group appears in at least one test set
    4. Not have overlap between train and test in same split
    """
    # Example: Custom logic here
    # For demonstration, this does a simple 80/20 split
    
    if random_seed is not None:
        random.seed(random_seed)
    
    shuffled = group_ids.copy()
    random.shuffle(shuffled)
    
    split_point = int(len(shuffled) * 0.8)
    train = shuffled[:split_point]
    test = shuffled[split_point:]
    
    # Return list of splits (even if just one)
    return [(train, test)]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_splits(
    splits: List[Tuple[List[str], List[str]]],
    group_ids: List[str],
    verbose: bool = True
) -> bool:
    """
    Validate that splits are properly formed.
    
    Checks:
    - No overlap between train and test in each split
    - All groups appear in at least one test set
    - No empty splits
    
    Parameters
    ----------
    splits : list of tuple
        List of (train_groups, test_groups) tuples
    group_ids : list of str
        List of all group identifiers
    verbose : bool, optional
        Whether to print validation details
        
    Returns
    -------
    valid : bool
        True if splits are valid, False otherwise
    """
    if verbose:
        print(f"\nValidating {len(splits)} splits...")
    
    all_valid = True
    tested_groups = set()
    
    for i, (train, test) in enumerate(splits):
        # Check no overlap
        overlap = set(train) & set(test)
        if overlap:
            print(f"  ✗ Split {i}: Train/test overlap: {overlap}")
            all_valid = False
        
        # Check not empty
        if not train or not test:
            print(f"  ✗ Split {i}: Empty {'train' if not train else 'test'} set")
            all_valid = False
        
        # Track tested groups
        tested_groups.update(test)
        
        if verbose:
            print(f"  Split {i}: {len(train)} train, {len(test)} test")
    
    # Check all groups tested
    untested = set(group_ids) - tested_groups
    if untested:
        print(f"  ✗ Groups never tested: {untested}")
        all_valid = False
    
    if verbose:
        if all_valid:
            print("  ✓ All splits valid!")
        print(f"  Coverage: {len(tested_groups)}/{len(group_ids)} groups tested")
    
    return all_valid


def compare_split_strategies(
    group_ids: List[str],
    strategies: dict,
    k: int = 5
):
    """
    Compare multiple split strategies side-by-side.
    
    Parameters
    ----------
    group_ids : list of str
        List of all group identifiers
    strategies : dict
        Dictionary mapping strategy_name -> split_function
    k : int, optional
        Parameter to pass to split functions
        
    Examples
    --------
    >>> groups = ['slice' + str(i) for i in range(10)]
    >>> strategies = {
    ...     'K-Fold': kfold_split,
    ...     'LOOCV': leave_one_out_split,
    ...     'Monte Carlo': monte_carlo_split
    ... }
    >>> compare_split_strategies(groups, strategies, k=5)
    """
    print(f"\nComparing split strategies on {len(group_ids)} groups:")
    print("=" * 60)
    
    for name, split_func in strategies.items():
        print(f"\n{name}:")
        try:
            splits = split_func(group_ids, k, random_seed=42)
            print(f"  Number of splits: {len(splits)}")
            
            train_sizes = [len(train) for train, test in splits]
            test_sizes = [len(test) for train, test in splits]
            
            print(f"  Train sizes: min={min(train_sizes)}, max={max(train_sizes)}, avg={sum(train_sizes)/len(train_sizes):.1f}")
            print(f"  Test sizes: min={min(test_sizes)}, max={max(test_sizes)}, avg={sum(test_sizes)/len(test_sizes):.1f}")
            
            # Check validity
            valid = validate_splits(splits, group_ids, verbose=False)
            print(f"  Valid: {'✓' if valid else '✗'}")
            
        except Exception as e:
            print(f"  Error: {e}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Examples of using different split functions.
    """
    
    # Example data
    group_ids = [f'slice{i:03d}' for i in range(20)]
    
    print("="*60)
    print("SPLIT FUNCTION EXAMPLES")
    print("="*60)
    
    # Example 1: K-fold
    print("\n1. K-Fold Split (k=5)")
    splits = kfold_split(group_ids, k=5, random_seed=42)
    validate_splits(splits, group_ids)
    
    # Example 2: LOOCV
    print("\n2. Leave-One-Out")
    splits = leave_one_out_split(group_ids, k=None, random_seed=42)
    print(f"  Generated {len(splits)} splits (one per group)")
    
    # Example 3: Single split
    print("\n3. Single Train/Test Split (80/20)")
    splits = single_train_test_split(group_ids, k=None, test_size=0.2)
    validate_splits(splits, group_ids)
    
    # Example 4: Monte Carlo
    print("\n4. Monte Carlo (10 iterations)")
    splits = monte_carlo_split(group_ids, k=None, n_iterations=10, test_size=0.2)
    print(f"  Generated {len(splits)} random splits")
    
    # Example 5: Compare strategies
    print("\n5. Strategy Comparison")
    strategies = {
        'K-Fold (k=5)': kfold_split,
        'K-Fold (k=10)': lambda g, k, s: kfold_split(g, 10, s),
        'LOOCV': leave_one_out_split,
        'Single Split': single_train_test_split,
    }
    compare_split_strategies(group_ids, strategies, k=5)