import unittest

import numpy as np
from sklearn.datasets import load_iris

from sude import sude


class TestSUDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = load_iris(return_X_y=True)[0][:20]

    def test_returns_expected_shape(self):
        embedding = sude(self.X, no_dims=2, k1=0, initialize="pca", T_epoch=2)

        self.assertEqual(embedding.shape, (20, 2))
        self.assertTrue(np.isfinite(embedding).all())

    def test_duplicate_rows_keep_duplicate_embeddings(self):
        X = np.vstack([self.X[:10], self.X[:3]])

        embedding = sude(X, no_dims=2, k1=0, initialize="pca", T_epoch=1)

        np.testing.assert_allclose(embedding[0], embedding[10])
        np.testing.assert_allclose(embedding[1], embedding[11])
        np.testing.assert_allclose(embedding[2], embedding[12])

    def test_supported_initializers(self):
        X = self.X[:15]

        for initialize in ("le", "pca", "mds"):
            with self.subTest(initialize=initialize):
                embedding = sude(
                    X,
                    no_dims=2,
                    k1=0,
                    initialize=initialize,
                    T_epoch=1,
                )
                self.assertEqual(embedding.shape, (15, 2))
                self.assertTrue(np.isfinite(embedding).all())

    def test_large_mode_runs(self):
        embedding = sude(
            self.X,
            no_dims=2,
            k1=0,
            initialize="pca",
            large=True,
            T_epoch=1,
        )

        self.assertEqual(embedding.shape, (20, 2))
        self.assertTrue(np.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
