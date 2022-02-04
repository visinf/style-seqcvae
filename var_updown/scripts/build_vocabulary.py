import argparse
import json
import os
from typing import Dict, List

from mypy_extensions import TypedDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Build a vocabulary out of COCO train2017 captions json file."
)

parser.add_argument(
    "-c",
    "--captions-jsonpath",
    default="data/coco/captions_train2017.json",
    help="Path to COCO train2017 captions json file.",
)
parser.add_argument("-t", "--word-count-threshold", type=int, default=5)
parser.add_argument(
    "-o",
    "--output-dirpath",
    default="data/vocabulary",
    help="Path to a (non-existent directory to save the vocabulary.",
)
parser.add_argument(
    "-s",
    "--senticap-jsonpath",
    default="data/SentiCap/data/senticap_dataset.json",
    help="Path to Senticap captions json file.",
)
parser.add_argument("-st", "--senticap-word-count-threshold", type=int, default=2)



# ------------------------------------------------------------------------------------------------
# All the punctuations in COCO captions, we will remove them.
# fmt: off
PUNCTUATIONS: List[str] = [
    "''", "'", "``", "`", "(", ")", "{", "}", ".", "?", "!", ",", ":", "-", "--", "...", ";"
]
# fmt: on

# Special tokens which should be added (all, or a subset) to the vocabulary.
# We use the same token for @@PADDING@@ and @@UNKNOWN@@ -- @@UNKNOWN@@.
SPECIAL_TOKENS: List[str] = ["@@UNKNOWN@@", "@@BOUNDARY@@"]

# Type for each COCO caption example annotation.
CocoCaptionExample = TypedDict("CocoCaptionExample", {"id": int, "image_id": int, "caption": str})
SentiCapExample = TypedDict("SentiCapExample", {"imgid": int, "sentences": Dict, "split": str, "filename": str})
# ------------------------------------------------------------------------------------------------


def build_caption_vocabulary(
    caption_json: List[CocoCaptionExample], senticap_json, word_count_threshold: int = 5, senticap_word_count_threshold: int = 2
) -> List[str]:
    r"""
    Given a list of COCO caption examples, return a list of unique captions tokens thresholded
    by minimum occurence.
    """

    word_counts: Dict[str, int] = {}
    word_counts2: Dict[str, int] = {}
    image_ids = set()

    # Accumulate unique caption tokens from all caption sequences.
    for item in tqdm(caption_json):
        image_ids.add(item["id"])
        caption: str = item["caption"].lower().strip()
        caption_tokens: List[str] = word_tokenize(caption)
        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]

        for token in caption_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    for item in tqdm(senticap_json):
        senti_coco_id = int(item["filename"].split(".")[0].split("_")[2])
        if(senti_coco_id in image_ids):
            for c in item["sentences"]:              
                caption: str = c["raw"].lower().strip()
                caption_tokens: List[str] = word_tokenize(caption)
                caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]
        
                for token in caption_tokens:
                    if token in word_counts2:
                        word_counts2[token] += 1
                    else:
                        word_counts2[token] = 1

    all_caption_tokens = [key for key in word_counts if word_counts[key] >= word_count_threshold]
    
    for key in word_counts2:
        count = word_counts2[key]
        
        if key in word_counts:
            count += word_counts[key]
            
        if(count >= senticap_word_count_threshold and key not in all_caption_tokens):
            all_caption_tokens.append(key)   
            
    caption_vocabulary: List[str] = sorted(list(all_caption_tokens))
    return caption_vocabulary


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Loading annotations json from {args.captions_jsonpath}...")
    captions_json = json.load(open(args.captions_jsonpath))["annotations"]
    print(f"Loading annotations json from {args.senticap_jsonpath}...")
    senticap_json = json.load(open(args.senticap_jsonpath))["images"]

    print("Building caption vocabulary...")
    caption_vocabulary: List[str] = build_caption_vocabulary(
        captions_json, senticap_json, args.word_count_threshold, args.senticap_word_count_threshold
    )
    caption_vocabulary = SPECIAL_TOKENS + caption_vocabulary
    print(f"Caption vocabulary size (with special tokens): {len(caption_vocabulary)}")

    # Write the vocabulary to separate namespace files in directory.
    print(f"Writing the vocabulary to {args.output_dirpath}...")
    print("Namespaces: tokens.")
    print("Non-padded namespaces: tokens.")

    os.makedirs(args.output_dirpath, exist_ok=True)

    with open(os.path.join(args.output_dirpath, "tokens.txt"), "w") as f:
        for caption_token in caption_vocabulary:
            f.write(caption_token + "\n")

    with open(os.path.join(args.output_dirpath, "non_padded_namespaces.txt"), "w") as f:
        f.write("tokens")
