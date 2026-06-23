"""
test_features.py — Tests para extracción y selección de features
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFeatureExtraction:
    """Tests para extraction.py"""

    def test_tfidf_vectorizer_builds(self):
        """build_tfidf_vectorizer debe retornar un TfidfVectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.features.extraction import build_tfidf_vectorizer
        vec = build_tfidf_vectorizer()
        assert isinstance(vec, TfidfVectorizer)

    def test_bow_vectorizer_builds(self):
        """build_bow_vectorizer debe retornar un CountVectorizer."""
        from sklearn.feature_extraction.text import CountVectorizer
        from src.features.extraction import build_bow_vectorizer
        vec = build_bow_vectorizer()
        assert isinstance(vec, CountVectorizer)

    def test_extract_features_shapes(self, sample_texts):
        """extract_features debe retornar matrices con shapes correctos."""
        from src.features.extraction import extract_features

        # Split manual para test
        X_train = np.array(sample_texts[:6])
        X_test  = np.array(sample_texts[6:])

        X_tr_vec, X_te_vec, vectorizer = extract_features(
            X_train, X_test,
            vectorizer_type="tfidf",
            max_features=100,
        )

        assert X_tr_vec.shape[0] == len(X_train)
        assert X_te_vec.shape[0] == len(X_test)
        assert X_tr_vec.shape[1] == X_te_vec.shape[1]  # mismo vocabulario

    def test_no_data_leakage(self, sample_texts):
        """El vectorizador debe ajustarse SOLO en train, no en test."""
        from src.features.extraction import build_tfidf_vectorizer

        X_train = np.array(sample_texts[:5])
        X_test  = np.array(sample_texts[5:])

        vec = build_tfidf_vectorizer(max_features=50)
        vec.fit(X_train)

        vocab_after_train = set(vec.vocabulary_.keys())

        # Transformar test NO debe cambiar el vocabulario
        vec.transform(X_test)
        vocab_after_test = set(vec.vocabulary_.keys())

        assert vocab_after_train == vocab_after_test, "Data leakage detectado"

    def test_tfidf_ngram_has_bigrams(self, sample_texts):
        """TF-IDF con ngram_range=(1,2) debe incluir bigramas."""
        from src.features.extraction import build_tfidf_ngram_vectorizer

        vec = build_tfidf_ngram_vectorizer(max_features=200)
        vec.fit(sample_texts)

        vocab = vec.vocabulary_
        has_bigram = any(" " in term for term in vocab.keys())
        assert has_bigram, "No se encontraron bigramas en el vocabulario"

    def test_invalid_vectorizer_type_raises(self, sample_texts):
        """extract_features debe lanzar ValueError con tipo inválido."""
        from src.features.extraction import extract_features

        X_train = np.array(sample_texts[:6])
        X_test  = np.array(sample_texts[6:])

        with pytest.raises(ValueError):
            extract_features(X_train, X_test, vectorizer_type="invalido")


class TestFeatureSelection:
    """Tests para selection.py"""

    def test_chi2_reduces_features(self, sample_texts, sample_intents):
        """chi2 debe reducir el número de features."""
        from src.features.extraction import extract_features
        from src.features.selection import select_chi2
        from sklearn.preprocessing import LabelEncoder

        X_train = np.array(sample_texts[:6])
        X_test  = np.array(sample_texts[6:])
        y_raw   = sample_intents[:6]

        le = LabelEncoder()
        y_train = le.fit_transform(y_raw)

        X_tr_vec, X_te_vec, _ = extract_features(
            X_train, X_test, vectorizer_type="tfidf", max_features=100
        )

        k = min(5, X_tr_vec.shape[1])
        X_tr_sel, X_te_sel, selector = select_chi2(X_tr_vec, y_train, X_te_vec, k=k)

        assert X_tr_sel.shape[1] <= X_tr_vec.shape[1]
        assert X_te_sel.shape[1] == X_tr_sel.shape[1]

    def test_variance_threshold_removes_constants(self):
        """VarianceThreshold debe eliminar features constantes."""
        from src.features.selection import select_variance_threshold
        from scipy.sparse import csr_matrix

        # Crear matriz con una columna constante
        X = csr_matrix(np.array([
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float))

        X_tr_sel, _, _ = select_variance_threshold(X, X, threshold=0.0)
        # La columna del medio (siempre 0) debe eliminarse
        assert X_tr_sel.shape[1] < X.shape[1]

    def test_invalid_selection_method_raises(self, sample_texts, sample_intents):
        """select_features debe lanzar ValueError con método inválido."""
        from src.features.extraction import extract_features
        from src.features.selection import select_features
        from sklearn.preprocessing import LabelEncoder

        X_train = np.array(sample_texts[:6])
        X_test  = np.array(sample_texts[6:])
        le = LabelEncoder()
        y_train = le.fit_transform(sample_intents[:6])

        X_tr_vec, X_te_vec, _ = extract_features(
            X_train, X_test, vectorizer_type="tfidf", max_features=50
        )

        with pytest.raises(ValueError):
            select_features(X_tr_vec, y_train, X_te_vec, method="invalido")