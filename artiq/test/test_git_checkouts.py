import os
import unittest
from tempfile import TemporaryDirectory

import pygit2

from artiq.master.experiments import GitBackend


class _GitBackendTestMixin:
    # Build commits directly from blobs and a tree. This needs no working tree,
    # so it works for both bare and non-bare repositories.
    bare = False

    def setUp(self):
        self.tmp_dir = TemporaryDirectory(prefix="artiq_experiments_test")
        self.repo = pygit2.init_repository(self.tmp_dir.name, bare=self.bare)
        signature = pygit2.Signature("Test", "test@example.com")
        self.revs = []
        for content in ["rev1", "rev2"]:
            blob = self.repo.create_blob(content.encode())
            tree_builder = self.repo.TreeBuilder()
            tree_builder.insert("experiment.py", blob, pygit2.GIT_FILEMODE_BLOB)
            tree = tree_builder.write()
            parents = [] if self.repo.head_is_unborn else [self.repo.head.target]
            self.revs.append(
                str(
                    self.repo.create_commit(
                        "HEAD", signature, signature, "commit", tree, parents
                    )
                )
            )

    def test_checkout_content(self):
        backend = GitBackend(self.tmp_dir.name)
        path, _, _ = backend.request_rev(self.revs[0])
        with open(os.path.join(path, "experiment.py")) as f:
            self.assertEqual(f.read(), "rev1")

    def tearDown(self):
        self.tmp_dir.cleanup()


class TestGitBackend(_GitBackendTestMixin, unittest.TestCase):
    bare = False

    def setUp(self):
        super().setUp()
        # Populate the working tree and index from HEAD so that status() is
        # clean before a checkout is requested.
        self.repo.checkout("HEAD", strategy=pygit2.GIT_CHECKOUT_FORCE)

    def test_checkout_leaves_index_alone(self):
        backend = GitBackend(self.tmp_dir.name)
        backend.request_rev(self.revs[0])
        self.assertEqual(self.repo.status(), {})


class TestGitBackendBare(_GitBackendTestMixin, unittest.TestCase):
    bare = True
