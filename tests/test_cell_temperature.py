"""Phase 2 checks. Run: .venv/bin/python -m unittest tests/test_cell_temperature.py"""
import unittest

from cell.temperature import cell_temperature


class CellTemperatureTest(unittest.TestCase):
    def test_strong_sun_heats_cell(self):
        # 1000 W/m² on a 32 °C day -> about 20.8 °C hotter
        self.assertAlmostEqual(cell_temperature(32, 1000), 52.8, places=1)

    def test_overcast_stays_near_air(self):
        self.assertLess(cell_temperature(28, 100) - 28, 3)

    def test_dark_equals_air(self):
        self.assertEqual(cell_temperature(25, 0), 25)


if __name__ == "__main__":
    unittest.main()
