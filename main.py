from tracker.byte_tracker import BYTETracker
from server import Khadas
import argparse


class Arguments:
    def __init__(self, track_thresh=0.5, track_buffer=30, mot20=False, match_thresh=0.7, aspect_ratio_thresh=1.6, min_box_area=10):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-show", action='store_true')
    return parser


if __name__ == '__main__':
    arg = parser().parse_args()
    khadas = Khadas()
    args = Arguments()
    khadas.tracker = BYTETracker(args, frame_rate=20)
    while True:
        if arg.show:
            khadas.show()
        else:
            khadas.tracking()
