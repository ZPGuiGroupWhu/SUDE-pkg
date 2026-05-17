import pathlib
import unittest


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPackagingMetadata(unittest.TestCase):
    def test_python_support_metadata_declares_open_ended_supported_versions(self):
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

        self.assertEqual(pyproject["project"]["requires-python"], ">=3.8")
        classifiers = pyproject["project"]["classifiers"]
        for version in ("3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(f"Programming Language :: Python :: {version}", classifiers)

    def test_default_dependencies_include_numba(self):
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

        dependencies = pyproject["project"]["dependencies"]

        self.assertTrue(
            any(dependency.startswith("numba") for dependency in dependencies),
            "numba should be installed by default with pip install sude",
        )

    def test_test_dependencies_include_tomli_for_legacy_python(self):
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

        test_dependencies = pyproject["project"]["optional-dependencies"]["test"]

        self.assertTrue(
            any(
                dependency.startswith("tomli")
                and 'python_version < "3.11"' in dependency
                for dependency in test_dependencies
            ),
            "Python < 3.11 needs tomli because tomllib is only in the stdlib on 3.11+",
        )

    def test_readme_documents_default_acceleration(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("Numba-accelerated kernels are installed by default", readme)
        self.assertNotIn("sude[accelerate]", readme)


if __name__ == "__main__":
    unittest.main()
