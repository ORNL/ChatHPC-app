import unittest

import python_project_template


class TestStringMethods(unittest.TestCase):
    def test_add1(self):
        for i in range(10):
            assert python_project_template.add1(i) == i + 1

    def test_main(self):
        python_project_template.add.main(["1.0"])


if __name__ == "__main__":
    unittest.main()
