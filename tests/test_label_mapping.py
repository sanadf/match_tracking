import unittest

from training.classes import map_label


class LabelMappingTest(unittest.TestCase):
    def test_required_football_label_synonyms(self):
        examples = {
            "Joueur": "player",
            "joueur": "player",
            "Football": "ball",
            "football": "ball",
            "football ball": "ball",
            "soccer ball": "ball",
            "Ballon": "ball",
            "ballon": "ball",
            "Gardien": "goalkeeper",
            "gardien": "goalkeeper",
            "gardien de but": "goalkeeper",
            "goalie": "goalkeeper",
            "keeper": "goalkeeper",
            "Arbitre": "referee",
            "arbitre": "referee",
            "ref": "referee",
            "referee": "referee",
            "match official": "referee",
        }

        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertEqual(map_label(raw).final_name, expected)


if __name__ == "__main__":
    unittest.main()
