import hashlib
import os
import pathlib
import shutil
import sys
import cv2
import numpy as np

#TODO
def add_path():
    Alphapose_path = os.path.abspath('joints_detectors/Alphapose')
    hrnet_path = os.path.abspath('joints_detectors/hrnet')
    trackers_path = os.path.abspath('pose_trackers')
    #paths = filter(lambda p: p not in sys.path, [Alphapose_path, hrnet_path, trackers_path])
    #sys.path.extend(paths)


def encode_for_videpose3d(boxes,keypoints,resolution, dataset_name):
	# Generate metadata:
	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	prepared_boxes = []
	prepared_keypoints = []
	for i in range(len(boxes)):
		if len(boxes[i]) == 0 or len(keypoints[i]) == 0:
			# No bbox/keypoints detected for this frame -> will be interpolated
			prepared_boxes.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
			prepared_keypoints.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
			continue

		prepared_boxes.append(boxes[i])
		prepared_keypoints.append(keypoints[i][:,:2])
		
	boxes = np.array(prepared_boxes, dtype=np.float32)
	keypoints = np.array(prepared_keypoints, dtype=np.float32)
	keypoints = keypoints[:, :, :2] # Extract (x, y)
	
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(boxes[:, 0])
	indices = np.arange(len(boxes))
	for i in range(4):
		boxes[:, i] = np.interp(indices, indices[mask], boxes[mask, i])
	for i in range(17):
		for j in range(2):
			keypoints[:, i, j] = np.interp(indices, indices[mask], keypoints[mask, i, j])
	
	print('{} total frames processed'.format(len(boxes)))
	print('{} frames were interpolated'.format(np.sum(~mask)))
	print('----------')
	
	return [{
		'start_frame': 0, # Inclusive
		'end_frame': len(keypoints), # Exclusive
		'bounding_boxes': boxes,
		'keypoints': keypoints,
	}], metadata

#call the fuction after getting the result from alphapose
def generate_kpts(final_result, args, video_name):
    kpts = []
    no_person = []
    for i in range(len(final_result)):
        if not final_result[i]['result']:  # No people
            no_person.append(i)
            kpts.append(None)
            continue

        #TODO: support multi-person via checking 'idx'    
        kpt = max(final_result[i]['result'],
                  key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

        kpts.append(kpt.data.numpy())

        for n in no_person:
            kpts[n] = kpts[-1]
        no_person.clear()

    for n in no_person:
        kpts[n] = kpts[-1] if kpts[-1] else kpts[n-1]

    name = f'{args.outputpath}/{video_name}.npz'
    kpts = np.array(kpts).astype(np.float32)
    #print('kpts npz save in ', name)
    #np.savez_compressed(name, kpts=kpts)

    # Generate metadata: TODO detect it from video
    resolution = {
				'w': 1920,
				'h': 1080,
			}


	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	output = {}
	dataset_name = "alphapose"
	output[dataset_name] = {}
	output[dataset_name]['custom'] = kpts]
	print('kpts npz save in ', name)
	np.savez_compressed(name, positions_2d=output, metadata=metadata)

    

    return kpts


def calculate_area(data):
    """
    Get the rectangle area of keypoints.
    :param data: AlphaPose json keypoint format([x, y, score, ... , x, y, score]) or AlphaPose result keypoint format([[x, y], ..., [x, y]])
    :return: area
    """
    data = np.array(data)

    if len(data.shape) == 1:
        data = np.reshape(data, (-1, 3))

    width = min(data[:, 0]) - max(data[:, 0])
    height = min(data[:, 1]) - max(data[:, 1])

    return np.abs(width * height)


def split_video(video_path):
    stream = cv2.VideoCapture(video_path)

    output_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_name = video_name[:video_name.rfind('.')]

    save_folder = pathlib.Path(f'./{output_dir}/alpha_pose_{video_name}/split_image/')
    shutil.rmtree(str(save_folder), ignore_errors=True)
    save_folder.mkdir(parents=True, exist_ok=True)

    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    length = len(str(total_frames)) + 1

    i = 1
    while True:
        grabbed, frame = stream.read()

        if not grabbed:
            print(f'Split totally {i + 1} images from video.')
            break

        save_path = f'{save_folder}/output{str(i).zfill(length)}.png'
        cv2.imwrite(save_path, frame)

        i += 1

    saved_path = os.path.dirname(save_path)
    print(f'Split images saved in {saved_path}')

    return saved_path