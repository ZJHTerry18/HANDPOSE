import os

READ_PATH = r'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\leap'
WRITE_PATH = r'D:\Workspace\HANDPOSE\dataset\test_type_new'
id_list = ['p1_fp']
hand_list = ['left','right']
SAMPLE_PER_HAND = 200

if not os.path.exists(WRITE_PATH):
    os.makedirs(WRITE_PATH)

index = 0
for id in id_list:
    for hand in hand_list:
        txt_files = os.listdir(os.path.join(READ_PATH, id, hand))
        with open(os.path.join(READ_PATH, 'valid_txt', id + '_' + hand + '.txt')) as f:
            valids1 = f.readlines()
        with open(os.path.join(READ_PATH, 'valid_txt_sort1', id + '_' + hand + '.txt')) as f:
            valids = f.readlines()
        for file in txt_files:
            if file[:-4] + '\n' in valids:
                namecomp = file.split('_')
                with open(os.path.join(READ_PATH, id, hand, file), 'r') as fr:
                    lines = fr.readlines()
                handid = '0' if hand == 'left' else '1'
                lines.insert(0, handid + '\n')
                seq = index * SAMPLE_PER_HAND + int(namecomp[1][:-4])
                newfilename = '_'.join([namecomp[0], handid, str(seq).zfill(4) + '.txt'])
                print(newfilename)
                with open(os.path.join(WRITE_PATH, newfilename), 'w') as fw:
                    fw.writelines(lines)
    index += 1
