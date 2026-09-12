"""Feature engineering helpers.

Implemented in a later phase. Phase 2 does not transform or encode features.
``Suspicious_Keyword`` must not be used as a model feature.
"""


def engineer_features(*_args, **_kwargs):
    raise NotImplementedError(
        "Feature engineering is planned for a later phase. "
        "Do not call this from Phase 2 validation."
    )
