import unittest

import chatkokkos
from chatkokkos.app import AppPreferences


class TestAppPreferences(unittest.TestCase):

    def test_create(self):
       preferences = AppPreferences()
       assert preferences.data_file == "/home/7ry/Data/ellora/kokkos-data/kokkos_create_context.json", "incorrect default data_file"

    def test_update(self):
       preferences = AppPreferences()
       preferences.data_file = 'new_file.json'
       assert preferences.data_file == "new_file.json", "incorrect default data_file"

if __name__ == "__main__":
    unittest.main()
