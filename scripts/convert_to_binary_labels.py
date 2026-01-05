#!/usr/bin/env python3
"""
Convert 8-class corner labels to binary shot/no-shot labels.

Mapping:
  SHOT: GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET
  NO_SHOT: CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE, CORNER_WON
"""
import json
import os
from pathlib import Path

# Class mapping
SHOT_CLASSES = {'GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET'}
NO_SHOT_CLASSES = {'CLEARED', 'NOT_DANGEROUS', 'FOUL', 'OFFSIDE', 'CORNER_WON'}

def convert_label(label: str) -> str:
    """Convert 8-class label to binary."""
    if label in SHOT_CLASSES:
        return 'SHOT'
    elif label in NO_SHOT_CLASSES:
        return 'NO_SHOT'
    else:
        raise ValueError(f"Unknown label: {label}")

def convert_labels_file(input_path: Path, output_path: Path):
    """Convert a Labels-ball.json file to binary format."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Track statistics
    shot_count = 0
    no_shot_count = 0

    # Convert each video's labels
    for video in data['videos']:
        for annotation in video['annotations'].get('anticipation', []):
            original_label = annotation['label']
            binary_label = convert_label(original_label)
            annotation['label'] = binary_label

            if binary_label == 'SHOT':
                shot_count += 1
            else:
                no_shot_count += 1

    # Save converted file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return shot_count, no_shot_count

def main():
    base_dir = Path('/home/mseo/CornerTactics/FAANTRA/data')
    input_dir = base_dir / 'corner_anticipation'
    output_dir = base_dir / 'corner_anticipation_binary'

    print("Converting corner labels to binary (SHOT/NO_SHOT)")
    print("=" * 60)

    total_shot = 0
    total_no_shot = 0

    for split in ['train', 'valid', 'test']:
        input_path = input_dir / split / 'Labels-ball.json'
        output_path = output_dir / split / 'Labels-ball.json'

        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue

        shot, no_shot = convert_labels_file(input_path, output_path)
        total_shot += shot
        total_no_shot += no_shot

        total = shot + no_shot
        shot_pct = shot / total * 100 if total > 0 else 0
        print(f"{split:6s}: {total:4d} clips | SHOT: {shot:4d} ({shot_pct:5.1f}%) | NO_SHOT: {no_shot:4d}")

    print("=" * 60)
    total = total_shot + total_no_shot
    shot_pct = total_shot / total * 100 if total > 0 else 0
    print(f"{'Total':6s}: {total:4d} clips | SHOT: {total_shot:4d} ({shot_pct:5.1f}%) | NO_SHOT: {total_no_shot:4d}")
    print(f"\nOutput saved to: {output_dir}")

if __name__ == '__main__':
    main()
