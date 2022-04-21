from easydict import EasyDict as edict

__C = edict()
cfg = __C

HANDPOSEDICT_26 = ["1 0 0 0 0", "0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1",
	"1 1 0 0 0", "0 1 1 0 0", 
	"1 0 1 0 0", "1 0 0 1 0", "1 0 0 0 1", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
	"1 1 1 0 0", "0 1 1 1 0", "0 0 1 1 1", 
	"1 1 0 1 0", "1 0 1 1 0", "1 0 0 1 1", "1 1 0 0 1",
	"0 1 1 1 1", "1 0 1 1 1",
	"1 1 0 1 1", "1 1 1 1 0", 
	"1 1 1 1 1"]
HANDPOSEDICT_10 = ["0 1 0 0 0", "1 1 0 0 0", "0 1 1 0 0", "0 1 0 1 0", "0 1 0 0 1",
 "1 1 1 0 0", "1 1 0 1 0", "1 1 0 0 1", "1 1 1 1 0", "1 1 1 1 1"]
HANDPOSEDICT_13 = ["0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1", "1 1 0 0 0",
 "0 1 1 0 0", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
 "0 1 1 1 0", "0 0 1 1 1", "0 1 1 1 1"]

### --------------------configs setting-------------------------- ###
## general settings
__C.HANDPOSE_DICT = HANDPOSEDICT_13
__C.NUM_POSE = 13

__C.FP_IMAGE_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\\fingerprint\p1'