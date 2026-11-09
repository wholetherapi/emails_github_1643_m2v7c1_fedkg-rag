class PrivacyPreservingAligner:
    """Aligns entities across graphs without revealing raw identifiers."""

    def __init__(self, rounds=10, clipping=1.0):
        self.rounds = rounds
        self.clipping = clipping

    def align(self, local_embeddings):
        aggregated = None
        for emb in local_embeddings:
            emb = emb / max(emb.norm(), self.clipping)
            aggregated = emb if aggregated is None else aggregated + emb
        return aggregated / max(len(local_embeddings), 1)
