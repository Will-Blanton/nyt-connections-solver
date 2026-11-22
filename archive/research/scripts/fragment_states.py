"""Experiment-only states with either full groups or one full group plus fragments."""
import itertools
import random

import torch
from torch.utils.data import Dataset


class FragmentStates(Dataset):
    """Twelve reproducible states per board/size, balanced over the sole target.

    Controls enumerate a designated target and one of three other-group choices.
    Their loss accepts all complete groups present, not just the designated target.
    Fragment states contain all three other groups, each strictly incomplete.
    Both arms use the same board frequency, target order and word permutations.
    Nothing here requires new embeddings or edge computations.
    """

    def __init__(self, full_boards, size, seed, *, fragments):
        if size not in (2, 3):
            raise ValueError('Fragment states require 8 or 12 words')
        self.boards = full_boards
        self.size = size
        self.seed = seed
        self.fragments = fragments
        self.candidates = torch.tensor(list(itertools.combinations(range(size * 4), 4)))

    def __len__(self):
        return len(self.boards) * 12

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        board_index, local = divmod(index, 12)
        target, choice = divmod(local, 3)
        board = self.boards[board_index]
        others = [g for g in range(4) if g != target]
        # Separate RNG streams keep word order identical across treatment/control.
        rng = random.Random(f'fragment-selection:{self.seed}:{index}')
        positions = list(range(4 * target, 4 * target + 4))
        for offset, group in enumerate(others):
            if self.fragments:
                count = (2 if offset == choice else 1) if self.size == 2 else (2 if offset == choice else 3)
            else:
                count = 4 if ((offset == choice) if self.size == 2 else (offset != choice)) else 0
            positions.extend(rng.sample(list(range(4 * group, 4 * group + 4)), count))
        permutation = list(range(self.size * 4))
        random.Random(f'fragment-order:{self.seed}:{index}').shuffle(permutation)
        positions = torch.tensor([positions[i] for i in permutation])
        group_ids = positions // 4
        complete = [g for g in range(4) if (group_ids == g).sum() == 4]
        candidate_groups = group_ids[self.candidates]
        relevance = (candidate_groups == candidate_groups[:, :1]).all(dim=1)
        if relevance.sum().item() != (1 if self.fragments else self.size):
            raise ValueError('Unexpected complete-group labels')
        return dict(
            x=board['x'][positions], edge_features=board['edge_features'][positions][:, positions],
            word_list=[board['word_list'][i] for i in positions.tolist()], date=board['date'],
            num_sets_remaining=self.size, source_group_idx=torch.tensor(complete),
            source_group_labels=[('purple', 'blue', 'green', 'yellow')[g] for g in complete],
            relevance=relevance, original_positions=positions, target_group=target,
        )
