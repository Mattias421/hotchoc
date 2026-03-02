#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
Modified to work with rVAD manifests by looking up text in the original LibriSpeech directory.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", help="The rVAD manifest TSV file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--original-libri", required=True,
                        help="Path to the ORIGINAL LibriSpeech root (e.g. /store/store4/data/LibriSpeech-dev-clean)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:

        # Skip the root line in the TSV
        root = next(tsv).strip()

        for line in tsv:
            line = line.strip()
            # line is like: 275156/8297-275156-0008.flac  101120

            # 1. Parse the filename to get IDs
            # We only care about the filename, not the folder path in the TSV
            file_path_parts = line.split()[0] # get the path part
            filename = os.path.basename(file_path_parts) # 8297-275156-0008.flac

            # ID format is usually: speaker-chapter-segment.flac
            # Example: 8297-275156-0008
            file_id_base = os.path.splitext(filename)[0]
            id_parts = file_id_base.split("-")

            if len(id_parts) < 3:
                print(f"Skipping malformed filename: {filename}")
                continue

            speaker_id = id_parts[0]
            chapter_id = id_parts[1]
            trans_filename = f"{speaker_id}-{chapter_id}.trans.txt"

            # 2. Construct path to the ORIGINAL transcript
            # LibriStructure: root/speaker/chapter/speaker-chapter.trans.txt
            original_trans_path = os.path.join(
                args.original_libri,
                speaker_id,
                chapter_id,
                trans_filename
            )

            # 3. Load Transcript (Memoization)
            # We use (speaker_id, chapter_id) as the key since that's where the text file is
            trans_key = f"{speaker_id}-{chapter_id}"

            if trans_key not in transcriptions:
                if not os.path.exists(original_trans_path):
                    print(f"Warning: Transcript not found at {original_trans_path}")
                    # Create empty dict so we don't crash, but won't output labels
                    transcriptions[trans_key] = {}
                else:
                    texts = {}
                    with open(original_trans_path, "r") as trans_f:
                        for tline in trans_f:
                            items = tline.strip().split()
                            # key: 8297-275156-0008, val: HELLO WORLD
                            texts[items[0]] = " ".join(items[1:])
                    transcriptions[trans_key] = texts

            # 4. Write Output
            # We look up the specific file_id (8297-275156-0008) in the loaded transcript
            current_trans_dict = transcriptions[trans_key]

            if file_id_base in current_trans_dict:
                text = current_trans_dict[file_id_base]

                # Write .wrd
                print(text)
                print(text, file=wrd_out)

                # Write .ltr (letters with | separator)
                print(
                    " ".join(list(text.replace(" ", "|"))) + " |",
                    file=ltr_out,
                )

if __name__ == "__main__":
    main()
