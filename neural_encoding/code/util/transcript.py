import pandas as pd
from pathlib import Path
from util.constants import FR

DEFAULT_COLS = ["word", "onset", "offset", "section"]


def get_transcript(lang: str, add_sentence_index: bool = False) -> pd.DataFrame:
    transcript_path = f"brain_encoding/ds003643-download/annotation/{lang}/lpp{lang}_word_information.csv"
    parse_tree_path = f"brain_encoding/ds003643-download/annotation/{lang}/lpp{lang}_tree.csv"

    if lang == FR:
        # ran into some parsing errors with the FR tree
        parse_tree_path = parse_tree_path.replace(".csv", "_fixed.csv")

    df_transcript = pd.read_csv(transcript_path, usecols=DEFAULT_COLS)

    if add_sentence_index:
        df_parse_tree = parse_trees_to_dataframe(parse_tree_path)
        df_transcript["sentence_index"] = df_parse_tree["sentence_index"]

    return df_transcript


def parse_trees_to_dataframe(file_path: str):
    import nltk

    data = []

    with open(file_path, "r") as fp:
        for sentence_index, line in enumerate(fp):
            tree_string = line.strip()
            try:
                tree = nltk.Tree.fromstring(tree_string)
                for word, pos in tree.pos():
                    data.append(
                        {"word": word, "pos": pos, "sentence_index": sentence_index}
                    )
            except ValueError:
                print(f"Failed to parse line {sentence_index + 1}")

    df = pd.DataFrame(data)
    return df


def get_aligned_transcript(lang: str, copy: bool = True) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[3]
    transcript_path = repo_root / "neural_encoding" / "mats" / "aligned-transcripts" / f"lpp{lang}_aligned_transcript.csv"
    transcript_df = pd.read_csv(
        transcript_path, usecols=DEFAULT_COLS + ["sentence_index"]
    )

    if copy:
        return transcript_df.copy()

    return transcript_df
