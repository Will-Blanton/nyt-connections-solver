"""Set/graph datasets: prepare original boards once and slice true partial states."""
import ast
import itertools
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from .edge_features.composer import EdgeFeatureComposer
from .candidate_features import slice_candidate_features


def load_connections(path):
    """Read the historical CSV using the baseline's existing data corrections.

    Preserve row order (Purple, Blue, Green, Yellow). April 1 variant puzzles
    were excluded from the baseline and remain excluded here.
    """
    frame = pd.read_csv(path, index_col=0)
    for row, value in {
        1298: "['line', 'plane', 'point', 'solid']",
        1892: "['abyss', 'fly', 'matrix', 'thing']",
    }.items():
        if row in frame.index:
            frame.loc[row, 'connections'] = value
    frame = frame[~frame['date'].astype(str).str.contains('04-01')].copy()
    frame['date'] = frame['date'].astype(str)
    frame['connections'] = frame['connections'].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
    return frame.reset_index(drop=True)


def collate_keep_wordlists(batch):
    return {key: [sample[key] for sample in batch] if key in {'word_list', 'source_group_labels'}
            else default_collate([sample[key] for sample in batch]) for key in batch[0]}


class ConnectionsDataset(Dataset):
    # Preserve the existing notebooks' metadata convention; these are not targets.
    GROUP_LABELS = ['yellow', 'green', 'blue', 'purple']

    def __init__(self, puzzle_df, word_emb_model=None, char_vectorizer=None,
                 num_sets_remaining=4, feature_builder=None, *, prepared_boards=None):
        if num_sets_remaining not in (1, 2, 3, 4):
            raise ValueError('num_sets_remaining must be between 1 and 4')
        self.num_sets_remaining = num_sets_remaining
        self.puzzle_list = []
        boards = prepared_boards if prepared_boards is not None else self.prepare(
            puzzle_df, word_emb_model, char_vectorizer, feature_builder)
        for board in boards:
            for selected in itertools.combinations(range(4), num_sets_remaining):
                positions = [4 * group + offset for group in selected for offset in range(4)]
                sample = {'x': board['x'][positions],
                          'word_list': [board['words'][i] for i in positions],
                          'date': board['date'], 'num_sets_remaining': num_sets_remaining,
                          'source_group_idx': torch.tensor(selected, dtype=torch.long),
                          'source_group_labels': [self.GROUP_LABELS[i] for i in selected]}
                if 'x_char' in board:
                    sample['x_char'] = board['x_char'][positions]
                if 'edge_features' in board:
                    sample['edge_features'] = board['edge_features'][positions][:, positions]
                if 'candidate_edge_features' in board:
                    sample['candidate_edge_features'] = slice_candidate_features(
                        board['candidate_edge_features'], positions, original_size=len(board['words']))
                self.puzzle_list.append(sample)

    @staticmethod
    def prepare(puzzle_df, word_emb_model, char_vectorizer, feature_builder, edge_builder=None):
        boards = []
        for date, groups in puzzle_df.groupby('date')['connections'].agg(list).items():
            if len(groups) != 4 or any(len(group) != 4 for group in groups):
                raise ValueError(f'Puzzle {date} must have four groups of four words')
            words = list(itertools.chain.from_iterable(groups))
            if len(set(words)) != 16 or any(not isinstance(w, str) for w in words):
                raise ValueError(f'Puzzle {date} must have 16 distinct strings')
            encoder = feature_builder if feature_builder is not None else word_emb_model
            with torch.inference_mode():
                x = torch.as_tensor(encoder.encode(words, convert_to_tensor=True)).detach().float().cpu()
            if x.ndim != 2 or x.shape[0] != 16 or not torch.isfinite(x).all():
                raise ValueError('Node builder must return finite (16, D) features')
            # clone outside inference mode: tensors must support training backward.
            board = {'date': date, 'words': words, 'x': x.clone()}
            if char_vectorizer is not None:
                board['x_char'] = torch.as_tensor(char_vectorizer.transform(words).toarray())
            if edge_builder is not None:
                board['edge_features'] = edge_builder.encode(words)
                if edge_builder.candidate_feature_channels:
                    board['candidate_edge_features'] = edge_builder.encode_candidates(words)
            boards.append(board)
        return boards

    def __len__(self):
        return len(self.puzzle_list)

    def __getitem__(self, index):
        return self.puzzle_list[index]


class GraphConnectionsDataset(ConnectionsDataset):
    def __init__(self, puzzle_df, word_emb_model=None, char_vectorizer=None,
                 num_sets_remaining=4, feature_builder=None, *, edge_providers=None,
                 cache=None, prepared_boards=None):
        self.edge_builder = (edge_providers if isinstance(edge_providers, EdgeFeatureComposer)
                             else EdgeFeatureComposer(edge_providers, cache=cache) if edge_providers is not None else None)
        if prepared_boards is None:
            if self.edge_builder is None:
                raise ValueError('Graph datasets require edge providers')
            prepared_boards = self.prepare(puzzle_df, word_emb_model, char_vectorizer, feature_builder, self.edge_builder)
        super().__init__(puzzle_df, word_emb_model, char_vectorizer, num_sets_remaining,
                         feature_builder, prepared_boards=prepared_boards)


def make_state_loaders(puzzle_df, word_embedder, vectorizer, batch_size, shuffle=True,
                       feature_builder=None, *, edge_providers=None, cache=None):
    """Materialize full boards once across all 2/3/4-group datasets."""
    builder = (edge_providers if isinstance(edge_providers, EdgeFeatureComposer)
               else EdgeFeatureComposer(edge_providers, cache=cache) if edge_providers is not None else None)
    boards = ConnectionsDataset.prepare(puzzle_df, word_embedder, vectorizer, feature_builder, builder)
    dataset_class = GraphConnectionsDataset if builder is not None else ConnectionsDataset
    datasets = {size: dataset_class(puzzle_df, num_sets_remaining=size, prepared_boards=boards)
                for size in (2, 3, 4)}
    loaders = {size: DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                               collate_fn=collate_keep_wordlists) for size, dataset in datasets.items()}
    return datasets, loaders
