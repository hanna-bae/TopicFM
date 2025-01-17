from src.config.default import _CN as cfg

TEST_BASE_PATH = "assets/scannet_test_1500"

cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
cfg.DATASET.TEST_DATA_ROOT = "data/scannet/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/scannet_test.txt"
cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"
cfg.DATASET.TEST_IMGSIZE = (640, 480)
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.MODEL.COARSE.N_SAMPLES = 0
cfg.MODEL.MATCH_COARSE.THR = 0.25
cfg.MODEL.LOSS.FINE_TYPE = 'sym_epi'
