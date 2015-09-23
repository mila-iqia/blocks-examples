#!/usr/bin/python

import argparse
import logging
import os
import subprocess
import tarfile
import urllib2
import uuid

from picklable_itertools.extras import equizip

TRAIN_DATA_URL = 'http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz'
VALID_DATA_URL = 'http://www.statmt.org/wmt15/dev-v2.tgz'
PREPROCESS_URL = 'https://raw.githubusercontent.com/lisa-groundhog/' +\
                 'GroundHog/master/experiments/nmt/preprocess/preprocess.py'
TOKENIZER_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder/' +\
                'master/scripts/tokenizer/tokenizer.perl'

TOKENIZER_PREFIXES = 'https://raw.githubusercontent.com/moses-smt/' +\
                     'mosesdecoder/master/scripts/share/nonbreaking_' +\
                     'prefixes/nonbreaking_prefix.'
BLEU_SCRIPT_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder' +\
                  '/master/scripts/generic/multi-bleu.perl'
OUTPUT_DIR = './data'
PREFIX_DIR = './share/nonbreaking_prefixes'

parser = argparse.ArgumentParser(
    description="""
This script donwloads parallel corpora given source and target pair language
indicators and preprocess it respectively for neural machine translation.For
the preprocessing, moses tokenizer is applied first then tokenized corpora
are used to extract vocabularies for source and target languages. Finally the
tokenized parallel corpora are shuffled for SGD.

Note that, this script is written specificaly for WMT15 training and
development corpora, hence change the corresponding sections if you plan to use
some other data.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-s", "--source", type=str, help="Source language",
                    default="cs")
parser.add_argument("-t", "--target", type=str, help="Target language",
                    default="en")
parser.add_argument("--source-dev", type=str, default="newstest2013.cs",
                    help="Source language dev filename")
parser.add_argument("--target-dev", type=str, default="newstest2013.en",
                    help="Target language dev filename")
parser.add_argument("--source-vocab", type=int, default=30000,
                    help="Source language vocabulary size")
parser.add_argument("--target-vocab", type=int, default=30000,
                    help="Target language vocabulary size")


def download_and_write_file(url, file_name):
    logger.info("Downloading [{}]".format(url))
    if not os.path.exists(file_name):
        path = os.path.dirname(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("...saving to: %s Bytes: %s" % (file_name, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % \
                (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,
        f.close()
    else:
        logger.info("...file exists [{}]".format(file_name))


def extract_tar_file_to(file_to_extract, extract_into, names_to_look):
    extracted_filenames = []
    try:
        logger.info("Extracting file [{}] into [{}]"
                    .format(file_to_extract, extract_into))
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames()
                         if any([ff.find(nn) > -1 for nn in names_to_look])]
        if not len(src_trg_files):
            raise ValueError("[{}] pair does not exist in the archive!"
                             .format(src_trg_files))
        for item in tar:
            # extract only source-target pair
            if item.name in src_trg_files:
                file_path = os.path.join(extract_into, item.path)
                if not os.path.exists(file_path):
                    logger.info("...extracting [{}] into [{}]"
                                .format(item.name, file_path))
                    tar.extract(item, extract_into)
                else:
                    logger.info("...file exists [{}]".format(file_path))
                extracted_filenames.append(
                    os.path.join(extract_into, item.path))
    except Exception as e:
        logger.error("{}".format(str(e)))
    return extracted_filenames


def tokenize_text_files(files_to_tokenize, tokenizer):
    for name in files_to_tokenize:
        logger.info("Tokenizing file [{}]".format(name))
        out_file = os.path.join(
            OUTPUT_DIR, os.path.basename(name) + '.tok')
        logger.info("...writing tokenized file [{}]".format(out_file))
        var = ["perl", tokenizer,  "-l", name.split('.')[-1]]
        if not os.path.exists(out_file):
            with open(name, 'r') as inp:
                with open(out_file, 'w', 0) as out:
                    subprocess.check_call(
                        var, stdin=inp, stdout=out, shell=False)
        else:
            logger.info("...file exists [{}]".format(out_file))


def create_vocabularies(tr_files, preprocess_file):
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.source))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.target))
    src_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.source)][0]]) + '.tok'
    trg_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.target)][0]]) + '.tok'
    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))
    return src_filename, trg_filename


def merge_parallel(src_filename, trg_filename, merged_filename):
    with open(src_filename, 'r') as left:
        with open(trg_filename, 'r') as right:
            with open(merged_filename, 'w') as final:
                for lline, rline in equizip(left, right):
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename):
    with open(merged_filename) as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename):
    logger.info("Shuffling jointly [{}] and [{}]".format(src_filename,
                                                         trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = str(uuid.uuid4())
    shuffled_filename = str(uuid.uuid4())
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        try:
            merge_parallel(src_filename, trg_filename, merged_filename)
            subprocess.check_call(
                " shuf {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg)
            logger.info(
                "...files shuffled [{}] and [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] and [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)


def main():
    train_data_file = os.path.join(OUTPUT_DIR, 'tmp', 'train_data.tgz')
    valid_data_file = os.path.join(OUTPUT_DIR, 'tmp', 'valid_data.tgz')
    preprocess_file = os.path.join(OUTPUT_DIR, 'preprocess.py')
    bleuscore_file = os.path.join(OUTPUT_DIR, 'multi-bleu.perl')
    tokenizer_file = os.path.join(OUTPUT_DIR, 'tokenizer.perl')
    source_prefix_file = os.path.join(PREFIX_DIR,
                                      'nonbreaking_prefix.' + args.source)
    target_prefix_file = os.path.join(PREFIX_DIR,
                                      'nonbreaking_prefix.' + args.target)

    # Download the News Commentary v10 ~122Mb and extract it
    download_and_write_file(TRAIN_DATA_URL, train_data_file)
    tr_files = extract_tar_file_to(
        train_data_file, os.path.dirname(train_data_file),
        ["{}-{}".format(args.source, args.target)])

    # Download development set and extract it
    download_and_write_file(VALID_DATA_URL, valid_data_file)
    val_files = extract_tar_file_to(
        valid_data_file, os.path.dirname(valid_data_file),
        [args.source_dev, args.target_dev])

    # Download bleu score calculation script
    download_and_write_file(BLEU_SCRIPT_URL, bleuscore_file)

    # Download preprocessing script
    download_and_write_file(PREPROCESS_URL, preprocess_file)

    # Download tokenizer
    download_and_write_file(TOKENIZER_URL, tokenizer_file)
    download_and_write_file(TOKENIZER_PREFIXES + args.source,
                            source_prefix_file)
    download_and_write_file(TOKENIZER_PREFIXES + args.target,
                            target_prefix_file)

    # Apply tokenizer
    tokenize_text_files(tr_files + val_files, tokenizer_file)

    # Apply preprocessing and construct vocabularies
    src_filename, trg_filename = create_vocabularies(tr_files, preprocess_file)

    # Shuffle datasets
    shuffle_parallel(os.path.join(OUTPUT_DIR, src_filename),
                     os.path.join(OUTPUT_DIR, trg_filename))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')

    args = parser.parse_args()
    main()
