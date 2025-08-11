# data/download.py
import os
import urllib.request

def download_tiny_shakespeare(target="data/raw/tiny_shakespeare.txt"):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists(target):
        urllib.request.urlretrieve(url, target)
        print("Downloaded tiny_shakespeare to", target)
    else:
        print("Already exists:", target)

if __name__ == "__main__":
    download_tiny_shakespeare()
