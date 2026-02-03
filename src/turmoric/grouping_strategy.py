"""
Flexible grouping strategies for train/test splits that prevent data leakage.

This module provides multiple strategies for grouping related images together
during train/test splitting, ensuring that data from the same biological unit
(slice, animal, subject, etc.) doesn't appear in both training and test sets.
"""

from pathlib import Path
from typing import List, Dict, Callable, Optional, Any
from collections import defaultdict
import re
from abc import ABC, abstractmethod


class GroupingStrategy(ABC):
    """
    Abstract base class for grouping strategies.
    
    A grouping strategy defines how to extract a group identifier from
    a filename to prevent data leakage during train/test splits.
    """
    
    @abstractmethod
    def extract_group_id(self, filepath: Path) -> str:
        """
        Extract the group identifier from a filepath.
        
        Parameters
        ----------
        filepath : Path
            Path to an image file
            
        Returns
        -------
        group_id : str
            Identifier for the biological group (slice, animal, etc.)
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of this strategy."""
        pass


class UnderscorePositionStrategy(GroupingStrategy):
    """
    Extract group ID from a specific position in underscore-delimited filename.
    
    Examples
    --------
    Filename: "group_treatment_slice123_quad1.tif"
    Position 2: "slice123"
    
    Filename: "experiment1_mouse5_region2_image.tif"  
    Position 1: "mouse5"
    """
    
    def __init__(self, position: int):
        """
        Initialize with the underscore-delimited position.
        
        Parameters
        ----------
        position : int
            Zero-indexed position in underscore-split filename
        """
        self.position = position
    
    def extract_group_id(self, filepath: Path) -> str:
        """Extract group ID from specified underscore position."""
        parts = filepath.stem.split('_')
        if len(parts) > self.position:
            return parts[self.position]
        else:
            raise ValueError(
                f"Filename '{filepath.name}' doesn't have enough underscore-separated "
                f"parts (need at least {self.position + 1}, found {len(parts)})"
            )
    
    def get_description(self) -> str:
        return f"Underscore position {self.position} (0-indexed)"


class RegexStrategy(GroupingStrategy):
    """
    Extract group ID using a regular expression pattern.
    
    Examples
    --------
    Pattern: r"slice(\d+)"
    Matches: "animal3_slice123_img1.tif" -> "123"
    
    Pattern: r"mouse_(\w+)_"
    Matches: "mouse_M456_region1.tif" -> "M456"
    
    Pattern: r"(Brain\d+)"
    Matches: "exp1_Brain042_section.tif" -> "Brain042"
    """
    
    def __init__(self, pattern: str, group: int = 1):
        """
        Initialize with regex pattern.
        
        Parameters
        ----------
        pattern : str
            Regular expression pattern with capture group(s)
        group : int, optional
            Which capture group to use (default: 1, the first group)
        """
        self.pattern = pattern
        self.group = group
        self.compiled_pattern = re.compile(pattern)
    
    def extract_group_id(self, filepath: Path) -> str:
        """Extract group ID using regex pattern."""
        match = self.compiled_pattern.search(filepath.name)
        if match:
            return match.group(self.group)
        else:
            raise ValueError(
                f"Pattern '{self.pattern}' not found in filename '{filepath.name}'"
            )
    
    def get_description(self) -> str:
        return f"Regex pattern: {self.pattern} (group {self.group})"


class MultiTokenStrategy(GroupingStrategy):
    """
    Combine multiple tokens from filename to create compound group ID.
    
    Useful when you need to group by animal AND region, for example.
    
    Examples
    --------
    Positions: [1, 2]
    Filename: "exp1_mouse5_hippocampus_img1.tif"
    Group ID: "mouse5_hippocampus"
    """
    
    def __init__(self, positions: List[int], delimiter: str = '_', join_with: str = '_'):
        """
        Initialize with multiple positions to combine.
        
        Parameters
        ----------
        positions : list of int
            Positions to extract and combine (0-indexed)
        delimiter : str, optional
            Character used to split filename (default: '_')
        join_with : str, optional
            Character to join extracted tokens (default: '_')
        """
        self.positions = sorted(positions)
        self.delimiter = delimiter
        self.join_with = join_with
    
    def extract_group_id(self, filepath: Path) -> str:
        """Extract and combine multiple tokens."""
        parts = filepath.stem.split(self.delimiter)
        
        max_pos = max(self.positions)
        if len(parts) <= max_pos:
            raise ValueError(
                f"Filename '{filepath.name}' doesn't have enough parts "
                f"(need at least {max_pos + 1}, found {len(parts)})"
            )
        
        tokens = [parts[pos] for pos in self.positions]
        return self.join_with.join(tokens)
    
    def get_description(self) -> str:
        return f"Combined positions {self.positions} with '{self.join_with}'"


class CustomFunctionStrategy(GroupingStrategy):
    """
    Use a custom function to extract group ID.
    
    Maximum flexibility - provide any function that takes a Path and returns a string.
    
    Examples
    --------
    >>> def my_custom_parser(filepath: Path) -> str:
    ...     # Extract animal ID from complex filename
    ...     name = filepath.stem
    ...     if 'mouse' in name.lower():
    ...         return name.split('mouse')[1].split('_')[0]
    ...     return name.split('_')[0]
    >>> 
    >>> strategy = CustomFunctionStrategy(my_custom_parser)
    """
    
    def __init__(self, func: Callable[[Path], str], description: str = "Custom function"):
        """
        Initialize with custom function.
        
        Parameters
        ----------
        func : callable
            Function that takes a Path and returns a group ID string
        description : str, optional
            Human-readable description of what the function does
        """
        self.func = func
        self.description = description
    
    def extract_group_id(self, filepath: Path) -> str:
        """Apply custom function to extract group ID."""
        return self.func(filepath)
    
    def get_description(self) -> str:
        return self.description


class RangeStrategy(GroupingStrategy):
    """
    Extract group ID from a range of underscore positions.
    
    Examples
    --------
    Range: (2, 4)
    Filename: "exp_cond_mouse123_slice5_img.tif"
    Group ID: "mouse123_slice5"
    """
    
    def __init__(self, start: int, end: int, delimiter: str = '_', join_with: str = '_'):
        """
        Initialize with start and end positions.
        
        Parameters
        ----------
        start : int
            Starting position (inclusive, 0-indexed)
        end : int
            Ending position (exclusive, like Python slicing)
        delimiter : str, optional
            Character used to split filename (default: '_')
        join_with : str, optional
            Character to join extracted tokens (default: '_')
        """
        self.start = start
        self.end = end
        self.delimiter = delimiter
        self.join_with = join_with
    
    def extract_group_id(self, filepath: Path) -> str:
        """Extract range of tokens."""
        parts = filepath.stem.split(self.delimiter)
        
        if len(parts) < self.end:
            raise ValueError(
                f"Filename '{filepath.name}' doesn't have enough parts "
                f"(need at least {self.end}, found {len(parts)})"
            )
        
        tokens = parts[self.start:self.end]
        return self.join_with.join(tokens)
    
    def get_description(self) -> str:
        return f"Range [{self.start}:{self.end}] with '{self.join_with}'"


class DirectoryBasedStrategy(GroupingStrategy):
    """
    Use parent directory structure for grouping.
    
    Useful when images are organized in directories by animal/slice.
    
    Examples
    --------
    Level: 0 (immediate parent)
    Path: /data/mouse123/images/slice1.tif
    Group ID: "images"
    
    Level: 1 (grandparent)
    Path: /data/mouse123/images/slice1.tif
    Group ID: "mouse123"
    """
    
    def __init__(self, level: int = 1):
        """
        Initialize with directory level.
        
        Parameters
        ----------
        level : int, optional
            How many levels up to use (0 = immediate parent, 1 = grandparent, etc.)
            Default: 1 (grandparent)
        """
        self.level = level
    
    def extract_group_id(self, filepath: Path) -> str:
        """Extract directory name at specified level."""
        try:
            # parents[0] is immediate parent, parents[1] is grandparent, etc.
            parent_dir = filepath.parents[self.level]
            return parent_dir.name
        except IndexError:
            raise ValueError(
                f"Path '{filepath}' doesn't have {self.level + 1} parent directories"
            )
    
    def get_description(self) -> str:
        level_names = ["parent", "grandparent", "great-grandparent"]
        level_name = level_names[self.level] if self.level < len(level_names) else f"level-{self.level}"
        return f"Directory-based ({level_name} directory)"


# =============================================================================
# HELPER FUNCTIONS FOR COMMON USE CASES
# =============================================================================

def create_strategy_from_example(
    example_filepath: str,
    group_id_in_example: str,
    delimiter: str = '_'
) -> GroupingStrategy:
    """
    Semi-automatically create a strategy by showing an example.
    
    This function tries to infer the best strategy based on an example
    filename and where the group ID appears in it.
    
    Parameters
    ----------
    example_filepath : str
        Example filename
    group_id_in_example : str
        The group ID that should be extracted from this example
    delimiter : str, optional
        Delimiter used in filename (default: '_')
        
    Returns
    -------
    strategy : GroupingStrategy
        Inferred strategy
        
    Examples
    --------
    >>> strategy = create_strategy_from_example(
    ...     "group_treatment_slice123_quad1.tif",
    ...     "slice123"
    ... )
    >>> # Returns: UnderscorePositionStrategy(2)
    
    >>> strategy = create_strategy_from_example(
    ...     "mouse_M456_region1_img.tif",
    ...     "M456"
    ... )
    >>> # Returns: UnderscorePositionStrategy(1)
    """
    filepath = Path(example_filepath)
    parts = filepath.stem.split(delimiter)
    
    # Try to find exact match in parts
    if group_id_in_example in parts:
        position = parts.index(group_id_in_example)
        print(f"Found '{group_id_in_example}' at position {position}")
        return UnderscorePositionStrategy(position)
    
    # Try to find as substring
    for i, part in enumerate(parts):
        if group_id_in_example in part:
            print(f"Found '{group_id_in_example}' as substring in position {i}: '{part}'")
            # Create regex to extract this pattern
            pattern = re.escape(group_id_in_example)
            return RegexStrategy(f"({pattern})")
    
    raise ValueError(
        f"Could not find '{group_id_in_example}' in '{example_filepath}'. "
        f"Parts found: {parts}"
    )


def test_strategy_on_files(
    strategy: GroupingStrategy,
    file_list: List[Path],
    print_results: bool = True
) -> Dict[str, List[Path]]:
    """
    Test a grouping strategy on a list of files to see how they're grouped.
    
    Parameters
    ----------
    strategy : GroupingStrategy
        Strategy to test
    file_list : list of Path
        Files to test on
    print_results : bool, optional
        Whether to print summary (default: True)
        
    Returns
    -------
    groups : dict
        Dictionary mapping group_id -> list of files
        
    Examples
    --------
    >>> files = list(Path("/data").glob("*.tif"))
    >>> strategy = UnderscorePositionStrategy(2)
    >>> groups = test_strategy_on_files(strategy, files)
    """
    groups = defaultdict(list)
    errors = []
    
    for filepath in file_list:
        try:
            group_id = strategy.extract_group_id(filepath)
            groups[group_id].append(filepath)
        except Exception as e:
            errors.append((filepath, str(e)))
    
    if print_results:
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy.get_description()}")
        print(f"{'='*60}")
        print(f"Total files: {len(file_list)}")
        print(f"Groups found: {len(groups)}")
        print(f"Errors: {len(errors)}")
        
        if groups:
            print(f"\nGroup sizes:")
            for group_id, files in sorted(groups.items()):
                print(f"  {group_id}: {len(files)} files")
        
        if errors:
            print(f"\nErrors encountered:")
            for filepath, error in errors[:5]:  # Show first 5
                print(f"  {filepath.name}: {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
    
    return dict(groups)


# =============================================================================
# PRE-DEFINED STRATEGIES FOR COMMON CASES
# =============================================================================

class CommonStrategies:
    """Collection of pre-defined strategies for common naming conventions."""
    
    @staticmethod
    def brain_slice_third_position() -> GroupingStrategy:
        """
        Standard TURMorIC format: group_treatment_sliceID_quad.tif
        Extracts sliceID from position 2.
        """
        return UnderscorePositionStrategy(2)
    
    @staticmethod
    def animal_id_second_position() -> GroupingStrategy:
        """
        Format: experiment_animalID_condition_image.tif
        Extracts animalID from position 1.
        """
        return UnderscorePositionStrategy(1)
    
    @staticmethod
    def mouse_number_regex() -> GroupingStrategy:
        """
        Matches patterns like 'mouse123', 'Mouse_456', 'M789'
        """
        return RegexStrategy(r'[Mm]ouse[_-]?(\d+)', group=1)
    
    @staticmethod
    def slice_number_regex() -> GroupingStrategy:
        """
        Matches patterns like 'slice123', 'Slice_456', 'S789'
        """
        return RegexStrategy(r'[Ss]lice[_-]?(\d+)', group=1)
    
    @staticmethod
    def subject_and_region() -> GroupingStrategy:
        """
        Combines subject ID (position 1) and brain region (position 2).
        Format: exp_subjectID_region_image.tif -> "subjectID_region"
        """
        return MultiTokenStrategy([1, 2])
    
    @staticmethod
    def by_parent_directory() -> GroupingStrategy:
        """
        Groups by immediate parent directory name.
        Useful when each animal has its own folder.
        """
        return DirectoryBasedStrategy(level=0)
    
    @staticmethod
    def by_grandparent_directory() -> GroupingStrategy:
        """
        Groups by grandparent directory name.
        Useful for structure like: /animal/session/images/
        """
        return DirectoryBasedStrategy(level=1)


# =============================================================================
# EXAMPLE USAGE AND DOCUMENTATION
# =============================================================================

if __name__ == "__main__":
    """
    Examples demonstrating different grouping strategies.
    """
    
    # Example 1: Simple underscore position
    print("Example 1: Underscore Position Strategy")
    print("-" * 60)
    strategy = UnderscorePositionStrategy(2)
    example_files = [
        Path("group1_control_slice123_quad1.tif"),
        Path("group1_control_slice123_quad2.tif"),
        Path("group1_control_slice456_quad1.tif"),
        Path("group2_treated_slice789_quad1.tif"),
    ]
    groups = test_strategy_on_files(strategy, example_files)
    
    # Example 2: Regex strategy
    print("\n\nExample 2: Regex Strategy")
    print("-" * 60)
    strategy = RegexStrategy(r'slice(\d+)')
    groups = test_strategy_on_files(strategy, example_files)
    
    # Example 3: Custom function
    print("\n\nExample 3: Custom Function Strategy")
    print("-" * 60)
    
    def extract_animal_code(filepath: Path) -> str:
        """Custom logic: extract animal code from complex naming."""
        name = filepath.stem
        # Look for pattern like 'M123' or 'Mouse456'
        if 'Mouse' in name or 'M' in name:
            parts = name.split('_')
            for part in parts:
                if part.startswith('M') or part.startswith('Mouse'):
                    return part
        return parts[0]
    
    strategy = CustomFunctionStrategy(
        extract_animal_code,
        description="Extract animal code (M### or Mouse###)"
    )
    
    custom_files = [
        Path("exp1_Mouse123_region1.tif"),
        Path("exp1_Mouse123_region2.tif"),
        Path("exp2_M456_hippocampus.tif"),
    ]
    groups = test_strategy_on_files(strategy, custom_files)
    
    # Example 4: Multi-token strategy
    print("\n\nExample 4: Multi-Token Strategy")
    print("-" * 60)
    strategy = MultiTokenStrategy([1, 2])  # Combine positions 1 and 2
    multi_files = [
        Path("exp_mouse1_hippo_img1.tif"),
        Path("exp_mouse1_hippo_img2.tif"),
        Path("exp_mouse1_cortex_img1.tif"),
        Path("exp_mouse2_hippo_img1.tif"),
    ]
    groups = test_strategy_on_files(strategy, multi_files)
    
    # Example 5: Using CommonStrategies
    print("\n\nExample 5: Pre-defined Common Strategies")
    print("-" * 60)
    print("Available strategies:")
    print("  - brain_slice_third_position()")
    print("  - animal_id_second_position()")
    print("  - mouse_number_regex()")
    print("  - slice_number_regex()")
    print("  - subject_and_region()")
    print("  - by_parent_directory()")
    print("  - by_grandparent_directory()")