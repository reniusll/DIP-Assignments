import argparse
from pathlib import Path


def write_split_list(split_dir: Path, output_file: Path) -> None:
    images = sorted(p for p in split_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    with output_file.open('w', encoding='utf-8') as f:
        for image_path in images:
            f.write(str(image_path.as_posix()) + '\n')
    print(f'Wrote {len(images)} entries to {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Generate train/val file lists for pix2pix-style datasets.')
    parser.add_argument('--dataset-root', required=True, help='Dataset root containing train/ and val/ folders.')
    parser.add_argument('--train-output', default='train_list.txt', help='Output txt file for training split.')
    parser.add_argument('--val-output', default='val_list.txt', help='Output txt file for validation split.')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError('Expected dataset_root to contain both train/ and val/ folders.')

    write_split_list(train_dir, Path(args.train_output))
    write_split_list(val_dir, Path(args.val_output))


if __name__ == '__main__':
    main()
