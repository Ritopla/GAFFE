from reference_extractor import AutoReferenceExtractor
import cv2
from video_filter import VideoFaceFilter


# Utilizzo
extractor = AutoReferenceExtractor()
reference_image, reference_encodings = extractor.extract_main_face_from_video("YW_WILTY_EP70_truth15.mp4")

# Salva la reference image per future analisi
cv2.imwrite("reference_face.jpg", reference_image)

videoFilter = VideoFaceFilter(reference_encodings)

new_frames = videoFilter.filter_video("YW_WILTY_EP70_truth15.mp4")

videoFilter.frames_to_video(new_frames, "VIDEO.mp4")


