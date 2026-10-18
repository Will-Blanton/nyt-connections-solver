"""Shared contract for providers that compute one or more edge channels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, Mapping, Any

if TYPE_CHECKING:
    from torch import Tensor


class EdgeFeatureProvider(Protocol):
    """A feature family with optional shared resources and cached computations.

    Implementations may retain dictionaries, encoders, and caches on the instance.
    Resources may also be shared between providers through their constructors.
    A single-feature provider uses the same interface with F = 1.
    """

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        """JSON-serializable algorithm version, configuration, and resource fingerprints.

        Bump the version whenever the calculation changes. Include model revision
        and lexical resource identity whenever those affect the output.
        """
        ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Nonempty, unique channel names in stable output order."""
        ...

    def encode(self, words: Sequence[str]) -> Tensor:
        """Return a CPU floating-point tensor of shape (N, N, F).

        N is len(words); F is len(feature_names). Each channel must be
        symmetric, finite, in [0, 1], and zero on the diagonal. Word order
        determines both edge axes, and channel order follows feature_names.
        Reordering words must reorder the corresponding rows and columns.

        Compute context-dependent features on the original puzzle, then slice
        both word axes for partial states. Providers must not use solution
        labels at inference time or mutate the supplied words.
        """
        ...
