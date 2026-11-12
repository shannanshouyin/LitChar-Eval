"""Download the PersoNet release used in the study."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
ROOT=Path(__file__).resolve().parents[1]
COMMIT='34745ded1ce168dcfb5ac7e5ca06fe364d16ff04'
EXPECTED='ecb37dc9d3f566836096a369ace2cc8182ee37ac971702822ade68b09509890d'
def main():
    target=ROOT/'data/raw/full_test_data.json';target.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/Gorov/personet_acl23/{COMMIT}/personet_data/full_test_data.json'
    with urlopen(url,timeout=120) as response: data=response.read()
    if hashlib.sha256(data).hexdigest()!=EXPECTED:
        raise ValueError('The downloaded corpus differs from the study release.')
    target.write_bytes(data)
    print('Saved data/raw/full_test_data.json')
if __name__=='__main__':main()
