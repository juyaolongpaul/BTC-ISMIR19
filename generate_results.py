import os
import mir_eval
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths_with_id, get_audio_paths
import argparse
import warnings
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if __name__ == '__main__':
# hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--audio_dir', type=str, default='./test')
    parser.add_argument('--src_csv', type=str, help='source csv path with song id', default='empty')
    parser.add_argument('--save_label_one_hot_dir', type=str, default='./test')
    parser.add_argument('--save_dir', type=str, default='./test')
    parser.add_argument('--save_attention_feature_dir', type=str, default='./test')
    parser.add_argument('--sep', type=str, default='\t', help='sep')
    args = parser.parse_args()

    config = HParams.load("run_config.yaml")
    if args.src_csv != 'empty':
        df = pd.read_csv(args.src_csv, index_col=False, sep=args.sep)
        songids = list(df['songid'].apply(str))  # obtain all the songs specified in csv that need to extract features
        audio_paths = get_audio_paths_with_id(args.audio_dir, songids)
    else:  # if not specified, list all the .wav and .mp3 files
        audio_paths = get_audio_paths(args.audio_dir)
    if args.voca is True:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = './test/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        model_file = './test/btc_model.pt'
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")

    model = BTC_model(config=config.model).to(device)

    # Load model
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file, map_location='cpu')
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.load_state_dict(checkpoint['model'])
        logger.info("restore model")



    # Chord recognition and save lab file
    for i, audio_path in enumerate(audio_paths):
        logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))
        # Load mp3
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        logger.info("audio file loaded and feature computation success : %s" % audio_path)

        # Majmin type chord recognition
        feature = feature.T
        feature = (feature - mean) / std
        time_unit = feature_per_second
        n_timestep = config.model['timestep']

        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        start_time = 0.0
        lines = []
        with torch.no_grad():
            model.eval()
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            for t in range(num_instance):

                self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                prediction, _ = model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                np_prediction = np.array(prediction)
                # turn it into a 2-d one hot matrix
                prediction_one_hot = [0] * 170
                prediction_one_hot[np_prediction[0]] = 1
                for each_int in np_prediction[1:]:
                    each_one_hot = [0] * 170
                    each_one_hot[each_int] = 1
                    prediction_one_hot = np.vstack((prediction_one_hot, each_one_hot))

                self_attn_output = self_attn_output.squeeze()
                np_self_attn_output = np.array(self_attn_output)
                if t == 0:
                    a_prediction_one_hot = prediction_one_hot
                    a_self_attn_output = np_self_attn_output
                else:
                    a_prediction_one_hot = np.vstack((a_prediction_one_hot, prediction_one_hot))
                    a_self_attn_output = np.vstack((a_self_attn_output, np_self_attn_output))
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        lines.append(
                            '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        break

        # lab file write
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.save_label_one_hot_dir):
            os.makedirs(args.save_label_one_hot_dir)
        if not os.path.exists(args.save_attention_feature_dir):
            os.makedirs(args.save_attention_feature_dir)
        save_path = os.path.join(args.save_dir,
                                 os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab')
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line)

        logger.info("label file saved : %s" % save_path)
        ## output feature
        feature_path_predicted_label = os.path.join(args.save_dir,
                                 os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '_predicted_one_hot_label.npy')
        with open(feature_path_predicted_label, 'wb') as f_predicted_label:
            np.save(f_predicted_label, a_prediction_one_hot)

        logger.info("label one hot predictions saved : %s" % feature_path_predicted_label)
        feature_path_attention_output = os.path.join(args.save_dir,
                                 os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '_attention_layer_output.npy')
        with open(feature_path_attention_output, 'wb') as f_attention_output:
            np.save(f_attention_output, a_self_attn_output)
        logger.info("attention embedding features saved : %s" % feature_path_attention_output)