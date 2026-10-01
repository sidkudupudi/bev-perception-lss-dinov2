from nuscenes.nuscenes import NuScenes

print("Initializing nuScenes Mini...")
nusc = NuScenes(version='v1.0-mini', dataroot='./data/sets/nuscenes'  # nuScenes v1.0-mini extracted here, verbose=True)

my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)

camera_channels = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

print("\n--- Synchronized Camera Feeds for Frame 0 ---")
for cam in camera_channels:
    cam_data = nusc.get('sample_data', my_sample['data'][cam])
    file_path = cam_data['filename']
    print(f"{cam}: {file_path}")