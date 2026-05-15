from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

class VideoEditor:
    @staticmethod
    def cut_clip(video_path: Path, start_time: float, end_time: float, output_path: Path):
        """Cuts a segment from the video."""
        with VideoFileClip(str(video_path)) as video:
            new_clip = video.subclipped(start_time, end_time)
            new_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
