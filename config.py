# -*- coding: utf-8 -*-

DEVICE = '0' # For GPU set device num which you want to use (or set 'cpu', but it's too slow)
#DEVICE = 'cpu'

# Our confidence for every person (bbox)
PERSON_CONF = 0.7

BAG_CONF = 0.7

IOU_THRESHOLD = 0.1

# It's xMem original config, you can try to change this values for your task (check xMem article)
XMEM_CONFIG = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 256,
    'min_mid_term_frames': 7,
    'max_mid_term_frames': 20,
    'max_long_term_elements': 10000,
}

# Max possible count of persons in video (if you has error, set bigger number)
MAX_OBJECT_CNT = 10

# Check new persons in frame every N frames
YOLO_EVERY = 60

# Resize processed video. For better results you can increase resolution
INFERENCE_SIZE = (960, 500)