import os

video_pathes = [
    'MOT17-03.mp4.cut.mp4',
 'MOT17-13.mp4.cut.mp4',
 'MOT17-10.mp4.cut.mp4',
 'MOT17-05.mp4.cut.mp4',
 'MOT17-06.mp4.cut.mp4',
 'MOT17-12.mp4.cut.mp4',
 'MOT17-14.mp4.cut.mp4',
 'MOT17-01.mp4.cut.mp4',
 'MOT17-11.mp4.cut.mp4',
 'MOT17-09.mp4.cut.mp4',
 'MOT17-02.mp4.cut.mp4',
 'MOT17-07.mp4.cut.mp4',
 'MOT17-08.mp4.cut.mp4',
 'MOT17-04.mp4.cut.mp4',
 'MOT16-03.mp4.cut.mp4',
 'MOT16-13.mp4.cut.mp4',
 'MOT16-10.mp4.cut.mp4',
 'MOT16-05.mp4.cut.mp4',
 'MOT16-06.mp4.cut.mp4',
 'MOT16-12.mp4.cut.mp4',
 'MOT16-14.mp4.cut.mp4',
 'MOT16-01.mp4.cut.mp4',
 'MOT16-11.mp4.cut.mp4',
 'MOT16-09.mp4.cut.mp4',
 'MOT16-02.mp4.cut.mp4',
 'MOT16-07.mp4.cut.mp4',
 'MOT16-08.mp4.cut.mp4',
 'MOT16-04.mp4.cut.mp4',
 'MOT20-01.mp4.cut.mp4',
 'MOT20-02.mp4.cut.mp4',
 'MOT20-03.mp4.cut.mp4',
 'MOT20-05.mp4.cut.mp4',
 'MOT20-04.mp4.cut.mp4',
 'MOT20-06.mp4.cut.mp4',
 'MOT20-07.mp4.cut.mp4',
 'MOT20-08.mp4.cut.mp4'
]

video_width_and_heights = [
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (640.0, 480.0),
    (640.0, 480.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (640.0, 480.0),
    (640.0, 480.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1920.0, 1080.0),
    (1173.0, 880.0),
    (1654.0, 1080.0),
    (1545.0, 1080.0),
    (1920.0, 734.0),
    (1920.0, 1080.0),
    (1920.0, 734.0)
]


def get_video_width_and_height(video_sequence):
    video_sequence = video_sequence[:8]
    if "{}.mp4.cut.mp4".format(video_sequence) not in video_pathes:
        raise NotImplementedError("{} not support now".format(video_sequence))
    else:
        return video_width_and_heights[video_pathes.index("{}.mp4.cut.mp4".format(video_sequence))]


def find_video_name(video_name):
    video_base_name = os.path.basename(video_name)
    parts = video_base_name.split('_')

    if len(parts) == 1:
        return video_base_name.split('.')[0]
    else:
        return parts[0]