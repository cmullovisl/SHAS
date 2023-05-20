from typing import Tuple
import argparse
from pathlib import Path

import numpy as np
import torch

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE
from data import FixedSegmentationDatasetNoTarget, segm_collate_fn
from models import SegmentationFrameClassifer, prepare_wav2vec
from segment import Segment, split_and_trim #, trim


def concat(sgm_a: Segment, sgm_b: Segment) -> Segment:
    probs_a = sgm_a.probs
    probs_b = sgm_b.probs
    filler = np.ones(sgm_b.start - sgm_a.end)
    # sgm = Segment(sgm_a.start, sgm_b.end, np.concatenate([probs_a, probs_b], 0))
    sgm = Segment(sgm_a.start, sgm_b.end, np.concatenate([probs_a, filler, probs_b], 0))

    assert sgm.end - sgm.start == len(sgm.probs)

    return sgm


def split(sgm: Segment, split_idx: int) -> tuple[Segment, Segment]:
    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)
    probs_b = sgm.probs[split_idx + 1:]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    assert (sgm_a.end - sgm_a.start == len(sgm_a.probs)
            and sgm_b.end - sgm_b.start == len(sgm_b.probs))

    return sgm_a, sgm_b


class OnlineSegmenter:

    def __init__(self, args):
        self.args = args

        device = (
            torch.device(f"cuda:0")
            if torch.cuda.device_count() > 0
            else torch.device("cpu")
        )
        self.device = device

        checkpoint = torch.load(args.path_to_checkpoint, map_location=device)

        # init wav2vec 2.0
        self.wav2vec_model = prepare_wav2vec(
            checkpoint["args"].model_name,
            checkpoint["args"].wav2vec_keep_layers,
            device,
        )
        # init segmentation frame classifier
        self.sfc_model = SegmentationFrameClassifer(
            d_model=HIDDEN_SIZE,
            n_transformer_layers=checkpoint["args"].classifier_n_transformer_layers,
        ).to(device)
        self.sfc_model.load_state_dict(checkpoint["state_dict"])
        self.sfc_model.eval()

        self.threshold = args.dac_threshold
        self.start = 0
        self.end = 0
        self.leftover = Segment(self.start, self.start, np.empty(0))
        self.probs = Segment(self.start, self.start, np.empty(0))

        self.min_segm_len = int(TARGET_SAMPLE_RATE * args.dac_min_segment_length)
        self.max_segm_len = int(TARGET_SAMPLE_RATE * args.dac_max_segment_length)

    def infer(
        self,
        wav2vec_model,
        sfc_model,
        data,
        duration_outframes,
        main_device,
    ) -> Tuple[np.array, np.array]:
        """Does inference with the Segmentation Frame Classifier for a single wav file

        Args:
            wav2vec_model: an instance of a wav2vec 2.0 model
            sfc_model: an instance of a segmentation frame classifier
            dataloader: a dataloader with the FixedSegmentationDataset of a wav
            main_device: the main torch.device

        Returns:
            Tuple[np.array, np.array]: the segmentation frame probabilities for the wav
                and (optionally) their ground truth values
        """

        talk_probs = []

        for audio, targets, in_mask, out_mask, included, starts, ends in iter(data):

            audio = audio.to(main_device)
            in_mask = in_mask.to(main_device)
            out_mask = out_mask.to(main_device)

            with torch.no_grad():
                wav2vec_hidden = wav2vec_model(
                    audio, attention_mask=in_mask
                ).last_hidden_state

                # some times the output of wav2vec is 1 frame larger/smaller
                # correct for these cases
                size1 = wav2vec_hidden.shape[1]
                size2 = out_mask.shape[1]
                if size1 != size2:
                    if size1 < size2:
                        out_mask = out_mask[:, :-1]
                        ends = [e - 1 for e in ends]
                    else:
                        wav2vec_hidden = wav2vec_hidden[:, :-1, :]

                logits = sfc_model(wav2vec_hidden, out_mask)
                probs = torch.sigmoid(logits)
                probs[~out_mask] = 0

            probs = probs.detach().cpu().numpy()
            talk_probs.append(probs)

#            # fill-in the probabilities and targets for the talk
#            for i in range(len(probs)):
#                start, end = starts[i], ends[i]
#                if included[i] and end > start:
#                    duration = end - start
#                    talk_probs[start:end] = probs[i, :duration]
#                elif not included[i]:
#                    talk_probs[start:end] = 0
#
#        # account for the rare incident that a frame didnt have a prediction
#        # fill-in those frames with the average of the surrounding frames
#        nan_idx = np.where(np.isnan(talk_probs))[0]
#        for j in nan_idx:
#            talk_probs[j] = np.nanmean(
#                talk_probs[max(0, j - 2) : min(duration_outframes, j + 3)]
#            )

        return talk_probs

    def pstrm(
        self,
        new_sgm: Segment,
    ) -> list[Segment]:
        """applies the probabilistic "Streaming" segmentation algorithm to split an audio
        into segments satisfying the max-segment-length and min-segment-length conditions

        Args:
            new_sgm (np.array): the binary frame-level probabilities
                output by the segmentation-frame-classifier

        Returns:
            Segment: resulting segmentation
        """


#        if not len(self.leftover.probs):
#            #self.leftover = Segment(self.start, self.start, np.empty(0))
#            self.leftover = Segment(new_sgm.start, new_sgm.start, np.empty(0))
#        end = new_sgm.end
#        current_sgm = concat(self.leftover, new_sgm)
        if len(self.leftover.probs):
            current_sgm = concat(self.leftover, new_sgm)
        else:
            current_sgm = new_sgm

        first_part, second_part = split(current_sgm, self.min_segm_len)

        if len(second_part.probs):
            sorted_indices = np.argsort(second_part.probs)
            split_idx = sorted_indices[0]
            min_prob = second_part.probs[split_idx]
        else:
            min_prob = 1.0

#        self.start = end

        # if min_prob < self.threshold and not_strict:
        if min_prob < self.threshold:
            first_part_b, self.leftover = split_and_trim(second_part, split_idx, self.threshold)
            if all(x < self.threshold for x in first_part.probs):
                # trim?
                if len(first_part_b.probs):
                    return first_part_b
            else:
                return concat(first_part, first_part_b)

        else:
            self.leftover = Segment(current_sgm.end, current_sgm.end, np.empty(0))
            return current_sgm

    def segment(self, wav_path):

        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path, self.args.inference_segment_length, self.args.inference_times
        )

        for inference_iteration in range(self.args.inference_times):
            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)

        for x in dataset:
            for s in self.process_audio(x, dataset.duration_outframes):
                yield s

    def process_audio(self, audio, duration_outframes):
        # get frame segmentation frame probabilities in the output space
        probs, = self.infer(
            self.wav2vec_model,
            self.sfc_model,
            [segm_collate_fn([audio])],
            duration_outframes,
            self.device,
        )
        probs = probs.squeeze()
        all_probs = np.concatenate((self.probs.probs, probs), 0)
        self.probs = Segment(self.probs.start, self.probs.end + len(probs), all_probs)
        assert (self.probs.end - self.probs.start) == len(self.probs.probs)

        while self.leftover.start + self.max_segm_len < self.probs.end:
            # this maked the above assertion fail (due to trimming?)
            # new_sgm, self.probs = split(self.probs, self.max_segm_len - len(self.leftover.probs))
            new_sgm, self.probs = split(self.probs, self.leftover.start + self.max_segm_len - self.probs.start)
            out_sgm = self.pstrm(new_sgm)
            if out_sgm is not None:
                yield out_sgm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="absolute path to the audio-frame-classifier checkpoint",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-wavs",
        type=str,
        help="absolute path to the directory of the wav audios to be segmented",
    )
    parser.add_argument(
        "--inference_batch_size",
        "-bs",
        type=int,
        default=12,
        help="batch size (in examples) of inference with the audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_segment_length",
        "-len",
        type=int,
        default=20,
        help="segment length (in seconds) of fixed-length segmentation during inference"
        "with audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_times",
        "-n",
        type=int,
        default=1,
        help="how many times to apply inference on different fixed-length segmentations"
        "of each wav",
    )
    parser.add_argument(
        "--dac_max_segment_length",
        "-max",
        type=float,
        default=18,
        help="the segmentation algorithm splits until all segments are below this value"
        "(in seconds)",
    )
    parser.add_argument(
        "--dac_min_segment_length",
        "-min",
        type=float,
        default=0.2,
        help="a split by the algorithm is carried out only if the resulting two segments"
        "are above this value (in seconds)",
    )
    parser.add_argument(
        "--dac_threshold",
        "-thr",
        type=float,
        default=0.5,
        help="after each split by the algorithm, the resulting segments are trimmed to"
        "the first and last points that corresponds to a probability above this value",
    )
    parser.add_argument(
        "--not_strict",
        action="store_true",
        help="whether segments longer than max are allowed."
        "If this argument is used, respecting the classification threshold conditions (p > thr)"
        "is more important than the length conditions (len < max)."
    )
    args = parser.parse_args()

    segmenter = OnlineSegmenter(args)

    for wav_path in sorted(list(Path(args.path_to_wavs).glob("*.wav"))):
        print(wav_path)
        for sgm in segmenter.segment(wav_path):
            # yaml_content = update_yaml_content(yaml_content, sgm, wav_path.name)
            #print(f'{sgm.start / TARGET_SAMPLE_RATE:.2f} {sgm.end / TARGET_SAMPLE_RATE:.2f}')
            #print(f'{sgm.start} {sgm.end}')
            print(f'{sgm.duration:.4f} {sgm.start / TARGET_SAMPLE_RATE:.4f}')
