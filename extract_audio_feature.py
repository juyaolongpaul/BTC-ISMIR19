import os
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features,  get_audio_paths_with_id, get_audio_paths
from multiprocessing import Pool, Process
import argparse
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def extract_audio_features(args, audio_path, i):
    # Chord recognition and save lab file
    logger.info("======== %d of %d in progress for feature extraction ========" % (i + 1, len(audio_paths)))
    # Load mp3
    if not os.path.exists(args.save_audio_feature):
        os.makedirs(args.save_audio_feature)
    audio_feature_path = os.path.join(args.save_audio_feature,
                                      os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav',
                                                                                                '') + '_chord_audio_feature.npy')
    if not (os.path.isfile(audio_feature_path) and args.reextract_features) == 'N':
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        with open(audio_feature_path, 'wb') as f_audio_feature:
            np.save(f_audio_feature, feature)
        logger.info("audio features saved : %s" % audio_path)


if __name__ == '__main__':
# hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, default='./test')
    parser.add_argument('--src_csv', type=str, help='source csv path with song id', default='empty')
    parser.add_argument('--save_audio_feature', type=str, default='./test')
    parser.add_argument('--sep', type=str, default='\t', help='sep')
    parser.add_argument('--need_multithreads', type=str, default='Y', help='specify whether multi-threading is needed')
    parser.add_argument('--n_thread', type=int, default=8, help='thread number')
    parser.add_argument('--reextract_features', type=str, default='Y', help='specify whether to re-extract audio features')
    args = parser.parse_args()

    config = HParams.load("run_config.yaml")
    if args.src_csv != 'empty':
        df = pd.read_csv(args.src_csv, index_col=False, sep=args.sep)
        songids = list(df['songid'].apply(str))  # obtain all the songs specified in csv that need to extract features
        audio_paths = get_audio_paths_with_id(args.audio_dir, songids)
    else:  # if not specified, list all the .wav and .mp3 files
        audio_paths = get_audio_paths(args.audio_dir)

    if args.need_multithreads == 'Y':
        # jobs = []
        pool = Pool(args.n_thread)
        for i, audio_path in enumerate(audio_paths):
            # p = Process(target=extract_audio_features, args=(args, audio_path, i))
            # jobs.append(p)
            # p.start()
            pool.apply_async(extract_audio_features, args=(args, audio_path, i))
        pool.close()
        pool.join()
    else:
        for i, audio_path in enumerate(audio_paths):
            extract_audio_features(args, audio_path, i)

