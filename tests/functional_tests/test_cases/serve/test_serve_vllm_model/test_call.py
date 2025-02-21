import unittest

import requests


class TestAPI(unittest.TestCase):
    def test_generate_endpoint(self):
        url = "http://127.0.0.1:6701/generate"
        headers = {"Content-Type": "application/json", "accept": "application/json"}
        test_data = {"prompt": "Introduce BAAI."}

        response = requests.post(url, headers=headers, json=test_data)

        self.assertEqual(
            response.status_code,
            200,
            f"Expected status code 200, got {response.status_code}. Response: {response}",
        )

        self.assertGreater(len(response.text), 0, "Generated text should not be empty")


if __name__ == "__main__":
    unittest.main()
