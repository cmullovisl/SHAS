from typing import Tuple
import argparse
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE
from data import FixedSegmentationDatasetNoTarget, segm_collate_fn
from eval import infer
from models import SegmentationFrameClassifer, prepare_wav2vec


@dataclass
class Segment:
    start: float
    end: float
    probs: np.array
    decimal: int = 4

    @property
    def duration(self):
        return float(round((self.end - self.start) / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset(self):
        return float(round(self.start / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset_plus_duration(self):
        return round(self.offset + self.duration, self.decimal)


def trim(sgm: Segment, threshold: float) -> Segment:
    """reduces the segment to between the first and last points that are above the threshold

    Args:
        sgm (Segment): a segment
        threshold (float): probability threshold

    Returns:
        Segment: new reduced segment
    """
    included_indices = np.where(sgm.probs >= threshold)[0]
    
    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, sgm.probs[i:j])

    return sgm


def split_and_trim(
    sgm: Segment, split_idx: int, threshold: float
) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments

    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        threshold (float): probability threshold

    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    sgm_a = trim(sgm_a, threshold)
    sgm_b = trim(sgm_b, threshold)

    return sgm_a, sgm_b


def update_yaml_content(
    yaml_content: list[dict], segments: list[Segment], wav_name: str
) -> list[dict]:
    """extends the yaml content with the segmentation of this wav file

    Args:
        yaml_content (list[dict]): segmentation in yaml format
        segments (list[Segment]): resulting segmentation from pdac
        wav_name (str): name of the wav file

    Returns:
        list[dict]: extended segmentation in yaml format
    """
    for sgm in segments:
        yaml_content.append(
            {
                "duration": sgm.duration,
                "offset": sgm.offset,
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )
    return yaml_content


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

        # sgm = Segment(0, len(probs), probs)
        # sgm = trim(sgm, args.dac_threshold)

        self.threshold = args.dac_threshold
        self.start = 0
        # self.start = sgm.start
        self.leftover = Segment(self.start, self.start, np.empty(0))

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

        # duration_outframes = dataloader.dataset.duration_outframes

#        talk_probs = np.empty(duration_outframes)
#        talk_probs[:] = np.nan
        talk_probs = []
        talk_targets = np.zeros(duration_outframes)

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
#                    if targets is not None:
#                        talk_targets[start:end] = targets[i, :duration].numpy()
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

        return talk_probs, talk_targets

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

        def _concat(sgm_a: Segment, sgm_b: Segment) -> Segment:
            probs_a = sgm_a.probs
            probs_b = sgm_b.probs
            sgm = Segment(sgm_a.start, sgm_b.end, np.concatenate([probs_a, probs_b], 0))
            return sgm

        def _split(sgm: Segment, split_idx: int) -> tuple[Segment, Segment]:
            probs_a = sgm.probs[:split_idx]
            sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)
            probs_b = sgm.probs[split_idx + 1:]
            sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)
            return sgm_a, sgm_b

        if not len(self.leftover.probs):
            self.leftover = Segment(self.start, self.start, np.empty(0))
        end = new_sgm.end
        current_sgm = _concat(self.leftover, new_sgm)

        first_part, second_part = _split(current_sgm, self.min_segm_len)
#        print(f'{self.start=}, {end=}')
#        print(f'{second_part.start=}, {second_part.end=}')
#        print(f'{self.leftover.start=}, {self.leftover.end=}')
#        print(f'{new_sgm.start=}, {new_sgm.end=}')
#        print(f'{current_sgm.duration=}, {current_sgm.start=}, {current_sgm.end=}')
#        print()

        if len(second_part.probs):
            sorted_indices = np.argsort(second_part.probs)
            split_idx = sorted_indices[0]
            min_prob = second_part.probs[split_idx]
        else:
            min_prob = 1.0

        self.start = end

        # if min_prob < self.threshold and not_strict:
        if min_prob < self.threshold:
            first_part_b, self.leftover = split_and_trim(second_part, split_idx, self.threshold)
            if all(x < self.threshold for x in first_part.probs):
                # trim?
                if len(first_part_b.probs):
                    return first_part_b
            else:
                return _concat(first_part, first_part_b)

        else:
            self.leftover = Segment(current_sgm.end, current_sgm.end, np.empty(0))
            return current_sgm

    def segment(self, wav_path):

        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path, self.args.inference_segment_length, self.args.inference_times
        )

        sgm_frame_probs = None
        for inference_iteration in range(self.args.inference_times):

            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
#            dataloader = DataLoader(
#                dataset,
#                batch_size=self.args.inference_batch_size,
#                num_workers=min(cpu_count() // 2, 4),
#                shuffle=False,
#                drop_last=False,
#                collate_fn=segm_collate_fn,
#            )
#
#        sgm_frame_probs /= self.args.inference_times
#
#        for s in self.process_audio(dataset):
#            yield s
        yield (s for s in self.process_audio(dataset))

    def process_audio(self, dataset):
        class _DataLoader:
            def __init__(self, data, duration_outframes):
                class _Dataset:
                    duration_outframes = 0
                self.dataset = _Dataset()
                self.dataset.duration_outframes = duration_outframes
                self.data = data

            def __iter__(self):
                # return iter(self.data)
                return (segm_collate_fn([x]) for x in self.data)

#        dataloader = _DataLoader(dataset, dataset.duration_outframes)
#
#        # get frame segmentation frame probabilities in the output space
#        probs, _ = infer(
#            self.wav2vec_model,
#            self.sfc_model,
#            dataloader,
#            self.device,
#        )
#        if sgm_frame_probs is None:
#            sgm_frame_probs = probs.copy()
#        else:
#            sgm_frame_probs += probs

        def _split(sgm: Segment, split_idx: int) -> tuple[Segment, Segment]:
            probs_a = sgm.probs[:split_idx]
            sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)
            probs_b = sgm.probs[split_idx + 1:]
            sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)
            return sgm_a, sgm_b

#        all_probs = []
#        for x in dataset:
#            # get frame segmentation frame probabilities in the output space
#            probs, _ = self.infer(
#                self.wav2vec_model,
#                self.sfc_model,
#                [segm_collate_fn([x])],
#                dataset.duration_outframes,
#                self.device,
#            )
#            all_probs += probs
#        probs = np.concatenate(all_probs, 1).squeeze()
#
#        sgm = Segment(0, len(probs), probs)
#        while self.start < len(probs):
#            end = min(self.leftover.start + self.max_segm_len, len(probs))
#            end_sgm, _ = _split(sgm, end - sgm.start)
#            _, new_sgm = _split(end_sgm, self.start - sgm.start)
#            out_sgm = self.pstrm(new_sgm)
#            if out_sgm is not None:
#                yield out_sgm

        all_probs = np.empty(0)
        for x in dataset:
            # get frame segmentation frame probabilities in the output space
            (probs,), _ = self.infer(
                self.wav2vec_model,
                self.sfc_model,
                [segm_collate_fn([x])],
                dataset.duration_outframes,
                self.device,
            )
            all_probs = np.concatenate((all_probs, probs.squeeze()), 0)

            sgm = Segment(0, len(all_probs), all_probs)
            while (end := self.leftover.start + self.max_segm_len) < len(all_probs):
                end_sgm, _ = _split(sgm, end - sgm.start)
                _, new_sgm = _split(end_sgm, self.start - sgm.start)
                out_sgm = self.pstrm(new_sgm)
                if out_sgm is not None:
                    yield out_sgm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_segmentation_yaml",
        "-yaml",
        type=str,
        required=True,
        help="absolute path to the yaml file to save the generated segmentation",
    )
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
    parser.add_argument(
        "--segmentation_method",
        type=str,
        default='pdac',
        choices=['pdac', 'pstrm'],
        help="Segmentation method to be used."
    )
    args = parser.parse_args()

    segmenter = OnlineSegmenter(args)

    yaml_content = []
    for wav_path in tqdm(sorted(list(Path(args.path_to_wavs).glob("*.wav")))):
        for sgm in segmenter.segment(wav_path):
            yaml_content = update_yaml_content(yaml_content, sgm, wav_path.name)

    path_to_segmentation_yaml = Path(args.path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)

    print(
        f"Saved SHAS segmentation with max={args.dac_max_segment_length} & "
        f"min={args.dac_min_segment_length} at {path_to_segmentation_yaml}"
    )

