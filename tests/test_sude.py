import unittest
from unittest import mock

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sude._sude as sude_module
import sude.learning as learning_module
from sude import SUDE, sude


class TestSUDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = load_iris(return_X_y=True)[0][:20]
        cls.X_query = np.vstack([cls.X[:4], cls.X[:2]])

    def assert_embeddings_equivalent(self, embedding_a, embedding_b):
        self.assertEqual(embedding_a.shape, embedding_b.shape)
        np.testing.assert_allclose(
            pdist(embedding_a),
            pdist(embedding_b),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_function_matches_estimator_fit_transform(self):
        model = SUDE(
            n_components=2,
            n_neighbors=0,
            init="pca",
            max_iter=2,
        )
        embedding_from_estimator = model.fit_transform(self.X)
        embedding_from_function = sude(
            self.X,
            n_components=2,
            n_neighbors=0,
            init="pca",
            max_iter=2,
        )

        np.testing.assert_allclose(
            embedding_from_estimator,
            embedding_from_function,
        )

    def test_fit_sets_expected_attributes(self):
        model = SUDE(
            n_components=2,
            n_neighbors=10,
            init="pca",
            max_iter=1,
        ).fit(self.X)

        self.assertEqual(model.embedding_.shape, (20, 2))
        self.assertEqual(model.Y_landmarks_.shape[1], 2)
        self.assertEqual(model.n_features_in_, self.X.shape[1])
        self.assertGreater(model.n_landmarks_, 0)
        self.assertEqual(model.get_feature_names_out().tolist(), ["sude0", "sude1"])

    def test_fit_caches_landmark_neighbor_index(self):
        model = SUDE(
            n_components=2,
            n_neighbors=10,
            init="pca",
            max_iter=1,
        ).fit(self.X)

        self.assertTrue(hasattr(model, "landmark_nn_"))

    def test_small_landmark_sets_disable_numba_by_default(self):
        model = SUDE(
            n_components=2,
            n_neighbors=10,
            init="pca",
            max_iter=1,
        ).fit(self.X)

        self.assertFalse(model.used_numba_)

    def test_numba_auto_policy_uses_global_thresholds(self):
        self.assertFalse(
            learning_module.should_use_numba(
                learning_module.NUMBA_AUTO_MIN_SAMPLES - 1,
                learning_module.NUMBA_AUTO_MIN_LANDMARKS,
            )
        )
        self.assertFalse(
            learning_module.should_use_numba(
                learning_module.NUMBA_AUTO_MIN_SAMPLES,
                learning_module.NUMBA_AUTO_MIN_LANDMARKS - 1,
            )
        )
        self.assertEqual(
            learning_module.should_use_numba(
                learning_module.NUMBA_AUTO_MIN_SAMPLES,
                learning_module.NUMBA_AUTO_MIN_LANDMARKS,
            ),
            learning_module.NUMBA_AVAILABLE,
        )

    def test_invalid_numba_thresholds_default_to_numba_when_available(self):
        original_min_samples = learning_module.NUMBA_AUTO_MIN_SAMPLES
        original_min_landmarks = learning_module.NUMBA_AUTO_MIN_LANDMARKS
        try:
            learning_module.NUMBA_AUTO_MIN_SAMPLES = 0
            learning_module.NUMBA_AUTO_MIN_LANDMARKS = "invalid"

            self.assertEqual(
                learning_module.should_use_numba(1, 1),
                learning_module.NUMBA_AVAILABLE,
            )
        finally:
            learning_module.NUMBA_AUTO_MIN_SAMPLES = original_min_samples
            learning_module.NUMBA_AUTO_MIN_LANDMARKS = original_min_landmarks

    def test_fit_numba_policy_uses_original_samples_and_landmarks(self):
        n_samples = learning_module.NUMBA_AUTO_MIN_SAMPLES
        n_landmarks = learning_module.NUMBA_AUTO_MIN_LANDMARKS
        X_unique = np.arange(n_samples * 3, dtype=float).reshape(n_samples, 3)
        inverse_indices = np.arange(n_samples)
        id_samp = np.arange(n_landmarks)
        get_knn = np.zeros((n_samples, 1), dtype=int)
        rnn = np.ones(n_samples)
        captured = {}

        def fake_learning(*args, **kwargs):
            captured["use_numba"] = kwargs["use_numba"]
            return (
                np.zeros((n_landmarks, 2)),
                1,
                {"used_numba": kwargs["use_numba"]},
            )

        with mock.patch.object(
            sude_module,
            "_prepare_training_data",
            return_value=(X_unique, inverse_indices, None),
        ), mock.patch.object(
            sude_module,
            "_validate_parameters",
            return_value="pca",
        ), mock.patch.object(
            sude_module,
            "_compute_landmarks",
            return_value=(get_knn, rnn, id_samp),
        ), mock.patch.object(
            sude_module,
            "learning",
            side_effect=fake_learning,
        ), mock.patch.object(
            sude_module,
            "opt_scale",
            return_value=np.ones((n_landmarks, 1)),
        ), mock.patch.object(
            sude_module,
            "_embed_with_landmarks",
            return_value=np.zeros((n_samples - n_landmarks, 2)),
        ):
            sude_module._fit_embedding(
                X=np.empty((1, 3)),
                n_components=2,
                n_neighbors=1,
                normalize=True,
                large=False,
                init="pca",
                agg_coef=1.2,
                max_iter=1,
            )

        self.assertEqual(captured["use_numba"], learning_module.NUMBA_AVAILABLE)

    def test_fit_numba_policy_rejects_small_landmark_count(self):
        n_samples = learning_module.NUMBA_AUTO_MIN_SAMPLES
        n_landmarks = learning_module.NUMBA_AUTO_MIN_LANDMARKS - 1
        X_unique = np.arange(n_samples * 3, dtype=float).reshape(n_samples, 3)
        inverse_indices = np.arange(n_samples)
        id_samp = np.arange(n_landmarks)
        get_knn = np.zeros((n_samples, 1), dtype=int)
        rnn = np.ones(n_samples)
        captured = {}

        def fake_learning(*args, **kwargs):
            captured["use_numba"] = kwargs["use_numba"]
            return (
                np.zeros((n_landmarks, 2)),
                1,
                {"used_numba": kwargs["use_numba"]},
            )

        with mock.patch.object(
            sude_module,
            "_prepare_training_data",
            return_value=(X_unique, inverse_indices, None),
        ), mock.patch.object(
            sude_module,
            "_validate_parameters",
            return_value="pca",
        ), mock.patch.object(
            sude_module,
            "_compute_landmarks",
            return_value=(get_knn, rnn, id_samp),
        ), mock.patch.object(
            sude_module,
            "learning",
            side_effect=fake_learning,
        ), mock.patch.object(
            sude_module,
            "opt_scale",
            return_value=np.ones((n_landmarks, 1)),
        ), mock.patch.object(
            sude_module,
            "_embed_with_landmarks",
            return_value=np.zeros((n_samples - n_landmarks, 2)),
        ):
            sude_module._fit_embedding(
                X=np.empty((1, 3)),
                n_components=2,
                n_neighbors=1,
                normalize=True,
                large=False,
                init="pca",
                agg_coef=1.2,
                max_iter=1,
            )

        self.assertFalse(captured["use_numba"])

    def test_numba_policy_is_not_an_estimator_parameter(self):
        params = SUDE().get_params()
        self.assertNotIn("numba_min_samples", params)
        self.assertNotIn("numba_min_landmarks", params)
        self.assertNotIn("use_numba", params)

    def test_numba_policy_can_be_configured_with_module_globals(self):
        original_min_samples = learning_module.NUMBA_AUTO_MIN_SAMPLES
        original_min_landmarks = learning_module.NUMBA_AUTO_MIN_LANDMARKS
        try:
            learning_module.NUMBA_AUTO_MIN_SAMPLES = 10
            learning_module.NUMBA_AUTO_MIN_LANDMARKS = 5

            self.assertEqual(
                learning_module.should_use_numba(10, 5),
                learning_module.NUMBA_AVAILABLE,
            )
        finally:
            learning_module.NUMBA_AUTO_MIN_SAMPLES = original_min_samples
            learning_module.NUMBA_AUTO_MIN_LANDMARKS = original_min_landmarks

    def test_returns_expected_shape(self):
        embedding = sude(
            self.X,
            n_components=2,
            n_neighbors=0,
            init="pca",
            max_iter=2,
        )

        self.assertEqual(embedding.shape, (20, 2))
        self.assertTrue(np.isfinite(embedding).all())

    def test_duplicate_rows_keep_duplicate_embeddings(self):
        X = np.vstack([self.X[:10], self.X[:3]])

        embedding = sude(
            X,
            n_components=2,
            n_neighbors=0,
            init="pca",
            max_iter=1,
        )

        np.testing.assert_allclose(embedding[0], embedding[10])
        np.testing.assert_allclose(embedding[1], embedding[11])
        np.testing.assert_allclose(embedding[2], embedding[12])

    def test_estimator_supported_initializers(self):
        X = self.X[:15]

        for init in ("le", "pca", "mds"):
            with self.subTest(init=init):
                embedding = SUDE(
                    n_components=2,
                    n_neighbors=0,
                    init=init,
                    max_iter=1,
                ).fit_transform(X)
                self.assertEqual(embedding.shape, (15, 2))
                self.assertTrue(np.isfinite(embedding).all())

    def test_estimator_rejects_spectral_initializer(self):
        with self.assertRaisesRegex(
            ValueError,
            "init must be one of {'le', 'pca', 'mds'}",
        ):
            SUDE(
                n_components=2,
                n_neighbors=0,
                init="spectral",
                max_iter=1,
            ).fit(self.X[:15])

    def test_transform_embeds_new_samples(self):
        model = SUDE(
            n_components=2,
            n_neighbors=10,
            init="pca",
            max_iter=1,
        ).fit(self.X)

        embedding = model.transform(self.X_query)

        self.assertEqual(embedding.shape, (6, 2))
        self.assertTrue(np.isfinite(embedding).all())
        np.testing.assert_allclose(embedding[0], embedding[4])
        np.testing.assert_allclose(embedding[1], embedding[5])

    def test_pipeline_compatibility(self):
        pipeline = make_pipeline(
            StandardScaler(),
            SUDE(
                n_components=2,
                n_neighbors=0,
                init="pca",
                max_iter=1,
                normalize=False,
            ),
        )
        embedding = pipeline.fit_transform(self.X)

        self.assertEqual(embedding.shape, (20, 2))
        self.assertTrue(np.isfinite(embedding).all())

    def test_function_supported_initializers_match_estimator(self):
        X = self.X[:15]

        for init in ("le", "pca", "mds"):
            with self.subTest(init=init):
                embedding = sude(
                    X,
                    n_components=2,
                    n_neighbors=0,
                    init=init,
                    max_iter=1,
                )
                self.assertEqual(embedding.shape, (15, 2))
                self.assertTrue(np.isfinite(embedding).all())

    def test_function_rejects_spectral_initializer(self):
        with self.assertRaisesRegex(
            ValueError,
            "init must be one of {'le', 'pca', 'mds'}",
        ):
            sude(
                self.X[:15],
                n_components=2,
                n_neighbors=0,
                init="spectral",
                max_iter=1,
            )

    def test_function_rejects_paper_style_parameter_names(self):
        with self.assertRaises(TypeError):
            sude(
                self.X,
                no_dims=2,
                k1=0,
                initialize="pca",
                T_epoch=1,
            )

    def test_large_mode_runs(self):
        embedding = sude(
            self.X,
            n_components=2,
            n_neighbors=0,
            init="pca",
            large=True,
            max_iter=1,
        )

        self.assertEqual(embedding.shape, (20, 2))
        self.assertTrue(np.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
