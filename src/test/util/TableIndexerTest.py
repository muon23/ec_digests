import os.path
import unittest

import pandas as pd

from util.TableIndexer import TableIndexer
import warnings


class TableIndexerTest(unittest.TestCase):
    test_data_root = f"../../../data/test/{__name__}"

    # Create sample city data
    city_data = {
        'city': [
            'New York', 'Tokyo', 'London', 'Paris', 'Singapore',
            'Dubai', 'Sydney', 'Mumbai', 'Shanghai', 'Toronto'
        ],
        'country': [
            'USA', 'Japan', 'UK', 'France', 'Singapore',
            'UAE', 'Australia', 'India', 'China', 'Canada'
        ],
        'population': [
            8400000, 37400000, 9000000, 2200000, 5700000,
            3300000, 5300000, 20400000, 27100000, 3000000
        ],
        'area_km2': [
            784, 2194, 1572, 105, 728,
            4114, 12368, 603, 6341, 630
        ],
        'timezone': [
            'UTC-5', 'UTC+9', 'UTC+0', 'UTC+1', 'UTC+8',
            'UTC+4', 'UTC+10', 'UTC+5:30', 'UTC+8', 'UTC-5'
        ]
    }

    # Create DataFrame
    city_df = pd.DataFrame(city_data)

    def test_basic(self):
        # LlamaIndex issue
        warnings.filterwarnings("ignore", message="Call to deprecated method get_doc_id")

        indexer = TableIndexer()
        indexer.insert(self.city_df, metadata_fields=["city"])

        answer, metadata = indexer.query("What's the timezone of Tokyo?")
        print(answer, metadata)

        self.assertEqual(answer, "UTC+9")
        self.assertIn({"city": "Tokyo"}, metadata.values())

        if not os.path.exists(self.test_data_root):
            os.makedirs(self.test_data_root)

        city_info_index = os.path.join(self.test_data_root, "city_info_index")
        indexer.save(city_info_index)

        indexer2 = TableIndexer.load(city_info_index)
        answer, metadata = indexer2.query("Name a city in India")
        print(answer, metadata)

        self.assertEqual(answer, "Mumbai")


if __name__ == '__main__':
    unittest.main()
