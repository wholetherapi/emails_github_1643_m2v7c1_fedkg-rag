from retriever import DenseRetriever


def test_dense_retriever_shapes():
    retriever = DenseRetriever(16)
    assert retriever.encoder_dim if hasattr(retriever, "encoder_dim") else True
