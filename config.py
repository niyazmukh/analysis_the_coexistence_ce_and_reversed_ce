"""Single canonical problem-name dictionary (iconic set)."""

from typing import Dict, Tuple

problem_names: Dict[Tuple[int, int], str] = {
    (1, 1): "8 ABC",
    (1, 2): "8 BCD",
    (2, 1): "22 ABC",
    (2, 2): "22 BCD",
    (3, 1): "8 ABC",
    (3, 2): "8 BCD",
    (3, 3): "22 ABC",
    (3, 4): "22 BCD",
    (3, 5): "140 ABC",
    (3, 6): "140 BCD",
    (4, 1): "8 ABC",
    (4, 2): "8 BCD",
    (4, 3): "22 ABC",
    (4, 4): "22 BCD",
    (4, 5): "140 ABC",
    (4, 6): "140 BCD",
    (5, 1): "8' ABC",
    (5, 2): "8' BCD",
}
