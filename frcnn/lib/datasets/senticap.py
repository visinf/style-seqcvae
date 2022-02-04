import json
from nltk.tokenize import word_tokenize


class SenticapReader(object):
    def __init__(self, senticap_jsonpath: str, train_split=False, val_split=False, test_split=False, sentiment=None):
        self.senticap_jsonpath = senticap_jsonpath
        
        with open(senticap_jsonpath) as senti:
            senticap_json = json.load(senti)["images"]
        # fmt: off
        # List of punctuations taken from pycocoevalcap - these are ignored during evaluation.
        PUNCTUATIONS: List[str] = [
            "''", "'", "``", "`", "(", ")", "{", "}",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"
        ]
        # fmt: on

        # List of (image id, caption) tuples.
        self._captions: List[Tuple[int, List[str]]] = []
        self._image_ids = set()
        self.sentiment = sentiment

        print(f"Tokenizing captions from {senticap_jsonpath}...")
        for item in senticap_json:
            senti_coco_id = int(item["filename"].split(".")[0].split("_")[2])
            split = item["split"]
            if(train_split and (split == "train") or
               (val_split and (split == "val")) or
               (test_split and (split == "test"))):
                cap_added = False
                for c in item["sentences"]:
                    if(not sentiment or sentiment == "pos" and c["sentiment"] == 1 or sentiment == "neg" and c["sentiment"] == 0):
                        cap_added = True
                        if(c["sentiment"] == 0):
                            c["sentiment"] = -1
                            
                        caption: str = c["raw"].lower().strip()
                        caption_tokens: List[str] = word_tokenize(caption)
                        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]
                        self._captions.append((senti_coco_id, caption_tokens, c["sentiment"]))
                if(cap_added):
                    self._image_ids.add(senti_coco_id)
                    
        
  

    def __len__(self):
        return len(self._captions)

    def __getitem__(self, index):
        return self._captions[index]