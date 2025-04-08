import unittest
import pandas as pd
from helpers.DataProcessing_helper_functions import split_dataframe

class TestDataProcessingHelperFunctions(unittest.TestCase):
    def test_split_dataframe(self):
        # Create a sample dataframe
        data = {
            'content': ["This is a test document. " * 10],
            'other_column': ["metadata_value"]
        }
        dataframe = pd.DataFrame(data)

        # Call split_dataframe with specific chunk_size and chunk_overlap
        chunks = split_dataframe(dataframe, chunk_size=50, chunk_overlap=10)

        # Assertions
        self.assertIsInstance(chunks, pd.DataFrame)
        self.assertTrue(len(chunks) > 1)
        self.assertIn('text', chunks.columns)
        self.assertIn('other_column', chunks.columns)
        # Ensure that the chunks are of the correct size
        for text in chunks['text']:
            self.assertTrue(40 <= len(text) <= 50)  # Considering overlap

if __name__ == '__main__':
    unittest.main()
