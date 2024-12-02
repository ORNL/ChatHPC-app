import unittest

import chatkokkos


class TestStringMethods(unittest.TestCase):
    def test_add1(self):
        for i in range(10):
            assert chatkokkos.add1(i) == i + 1

    def test_main(self):
        chatkokkos.add.main(["1.0"])


if __name__ == "__main__":
    unittest.main()
