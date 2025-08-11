import os
from nanoKimi.tokenizer import build_tokenizer

def test_tokenizer_build(tmp_path):
    cfg = {
        "type": "sentencepiece",
        "vocab_size": 100,
        "model_prefix": str(tmp_path / "test_tok")
    }

    # Build tokenizer
    tokenizer = build_tokenizer(cfg, train_texts=["hello world", "nanoKimi rocks!"])
    assert tokenizer.vocab_size() >= 2

    # Save and reload
    save_path = tmp_path / "tokenizer.model"
    tokenizer.save(str(save_path))
    assert os.path.exists(save_path)
