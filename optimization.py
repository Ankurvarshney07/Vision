import requests
import json
from deepdiff import DeepDiff
BASE_URL = "http://127.0.0.1:8000/"  # Replace with actual base URL
ENDPOINTS = [
   "api/1/institution/institution-online-degrees",
    # "/api/1/institution/institution-overview",
    # "/api/1/institution/institution-courses",
    # "/api/1/institution/institution-accepted-exam",
]
COLLEGE_IDS = [1, 115, 39711, 3, 5, 12, 13, 14, 15, 18, 21, 22, 23, 43, 52, 55, 63, 71, 82, 90, 96, 103, 104, 118, 125, 127, 1271, 1274, 1302, 1306, 1322, 1405, 157, 158, 160, 165, 186, 200, 206, 210, 229, 278, 303, 353, 375, 390, 484, 492, 524, 533, 537, 583, 588, 590, 591, 593, 599, 600, 602, 603, 604, 605, 609, 610, 621, 622, 641, 678, 679, 702, 711, 727, 733, 778, 791, 820, 848, 849, 850, 851, 986, 1590, 1613, 1758, 1765, 1766, 1811, 1812, 19745, 20728, 22329, 22343, 2449, 24710, 25506, 25797, 26043, 26057, 27384, 27413, 27436, 27854, 27944, 28588, 30833, 32140, 40943, 41028, 4357, 44448, 44620, 44667, 47495, 48012, 50272, 51462, 51528, 5160, 5197, 52419, 52633, 5490, 5504, 55593, 56832, 58299, 5818]
HEADERS = {
    "x-api-key": "xeJJzhaj1mQ-ksTB_nF_iH0z5YdG50yQtwQCzbcHuKA",
}
def fetch_response(endpoint, college_id, version):
    """Fetch response from API with given version"""
    params = {}
    params['college_id'] = college_id
    params["upgrade"] = version
    params['flag'] = 1
    response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    return response.json() if response.status_code == 200 else None
def test_api_responses():
    discrepancies = {}
    for endpoint in ENDPOINTS:
        for college_id in COLLEGE_IDS:
            old_response = fetch_response(endpoint, college_id, "old")
            new_response = fetch_response(endpoint, college_id, "new")
            if old_response and new_response:
                diff = DeepDiff(old_response, new_response, ignore_order=True)
                if diff:
                    discrepancies[endpoint] = diff
                    print('issue in ...')
                    print('endpoint: ', endpoint)
                    print('college_id: ', college_id)
                    print('\n')
            print('done checking => endpoint: ', endpoint, ' college_id: ',college_id)
    if discrepancies:
        with open("api_differences.json", "w") as f:
            json.dump(discrepancies, f, indent=4)
        print("API responses have differences. Check api_differences.json for details.")
test_api_responses()