import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import requests

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_engine import search_internet, search_with_fallback


class TestSearchEngine(unittest.TestCase):
    @patch('utils.search_engine.requests.get')
    def test_search_internet_success(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'Results': [
                {'Text': 'Result 1', 'FirstURL': 'https://example.com/1', 'Abstract': 'Abstract 1'},
                {'Text': 'Result 2', 'FirstURL': 'https://example.com/2', 'Abstract': 'Abstract 2'},
            ],
            'RelatedTopics': [
                {'Text': 'Topic 1', 'FirstURL': 'https://example.com/topic1', 'Abstract': 'Topic Abstract 1'},
                {'Topics': [{'Text': 'Subtopic'}]},  # This should be skipped
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the function
        results = search_internet('test query', max_results=3)
        
        # Verify the results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['title'], 'Result 1')
        self.assertEqual(results[0]['url'], 'https://example.com/1')
        self.assertEqual(results[0]['snippet'], 'Abstract 1')
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['params']['q'], 'test query')
        
    @patch('utils.search_engine.requests.get')
    def test_search_internet_retry(self, mock_get):
        # Create a mock response that will succeed on the second try
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {
            'Results': [{'Text': 'Result', 'FirstURL': 'https://example.com', 'Abstract': 'Abstract'}]
        }
        
        # Set up the mock to raise an exception on first call, then return the successful response
        mock_get.side_effect = [
            requests.exceptions.RequestException("Connection error"),
            mock_response_success
        ]
        
        # Call the function with retry capability
        with patch('utils.search_engine.time.sleep') as mock_sleep:  # Mock sleep to speed up tests
            results = search_internet('retry test', max_retries=2)
        
        # Verify we got results after retrying
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Result')
        
        # Verify the API was called twice
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)  # Sleep should have been called once
        
    @patch('utils.search_engine.search_internet')
    def test_search_with_fallback(self, mock_search):
        # Test when primary search returns results
        mock_search.return_value = [{'title': 'Test', 'url': 'https://test.com', 'snippet': 'Test snippet'}]
        results = search_with_fallback('test query')
        self.assertEqual(len(results), 1)
        
        # Test when primary search returns empty results
        mock_search.return_value = []
        results = search_with_fallback('test query')
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main() 