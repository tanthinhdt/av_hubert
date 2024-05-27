import os
import logging
import argparse
import pandas as pd
from pathlib import Path
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create vocab file for the dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--metadata-dir',
        type=str,
        help='Directory containing the dataset.',
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=2000,
        help='Size of the vocabulary.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory containing the dataset.',
    )
    return parser.parse_args()


def create_vocab(text: str, vocab_size: int, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    vocab_dir = (output_dir / f'spm_{vocab_size}').absolute()
    vocab_dir.mkdir(exist_ok=True)
    spm_filename_prefix = f'spm_unigram_{vocab_size}'

    with NamedTemporaryFile(mode='w', encoding='utf-8') as f:
        f.write(text)
        gen_vocab(
            Path(f.name), vocab_dir / spm_filename_prefix, 'unigram', vocab_size
        )

    vocab_path = (vocab_dir / spm_filename_prefix).as_posix() + '.txt'
    os.rename(vocab_path, f'{output_dir}/dict.wrd.txt')


def main(args: argparse.Namespace) -> None:
    logging.info(f'Creating vocabulary of size {args.vocab_size}')

    metadata_df = pd.concat([
        pd.read_parquet(metadata_file)
        for metadata_file in Path(args.metadata_dir).rglob('*.parquet')
    ])
    logging.info(f'Loaded metadata files with {len(metadata_df)} samples')

    text = '\n'.join(metadata_df['transcript'].str.lower().to_list()) + '\n'
    create_vocab(
        text=text,
        vocab_size=args.vocab_size,
        output_dir=Path(args.output_dir),
    )
    logging.info('Vocabulary created successfully')


if __name__ == '__main__':
    args = get_args()
    main(args=args)
