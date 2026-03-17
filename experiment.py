import labrad
import os
import time
import tifffile
import numpy as np
import labrad
import time
import socket
import PySpin
from labrad.server import LabradServer, setting
import base64
import pickle
import zlib
from datetime import datetime
import re


"""connect to servers"""
cxn = labrad.connect()
ws8 = cxn.ws8_ad3_lock_test
print("ws8 server connected!")
cam = cxn.FlirCameraTest
print("camera server connected!")

"""camera parameters and saving folder"""
# iter = 1
iter = 1

serial = "24053174"
exposure_time = 0.015
gain = 0
time_out_ms = 100
num_shots = 10

wavelength = '556' # '556' or '399'

def get_experiment_folder(base_dir, path=None):
    """
    Returns the folder name for saving data.
    If path is None, creates a new experimentN folder.
    If path is specified, uses that folder under base_dir.
    """
    if path is None:
        existing_nums = []
        if os.path.isdir(base_dir):
            for name in os.listdir(base_dir):
                match = re.fullmatch(r"experiment(\d+)", name)
                if match and os.path.isdir(os.path.join(base_dir, name)):
                    existing_nums.append(int(match.group(1)))
        next_num = max(existing_nums, default=0) + 1
        folder_name = f"experiment{next_num}"
    else:
        folder_name = path
    print("Data will be saved to folder:", folder_name)
    return folder_name

# 用法举例
base_dir = "C:/Users/lanty/Desktop/experiments"
path = None  # 新建 experimentN 文件夹
# path = "Trash"  # 存到 Trash 文件夹
# path = "experiment177"  # 存到已存在的 experiment177 文件夹
folder_name = get_experiment_folder(base_dir, path=path)

"""experiment parameters"""

central_556 = 539.386589 # THz
# delta_freq_list = np.arange(-700, 700, 20) # MHz
# delta_freq_list = [50, 20, 0, -10, -20, -30, -40, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100, -110, -120, -130, -140, -150, -160, -180, -200, -220, -240, -260, -280, -300]
# delta_freq_list_556 = [30, 20, 10, 0, -10, -20, -30, -40, -50, -55, -60, -65,-70,-75, -80,-85,-90,-95, -100, -105, -110, -130,-150, -200] # MHz
delta_freq_list_556 = [100, 80, 60, 40, 20, 0, -20, -40, -60, -80, -100] # MHz
# delta_freq_list_556 = np.arange(0,-33,-3) # MHz

# delta_freq_list_556 = [-100]  # MHz
# delta_freq_list_556 = np.arange(-200, 0, 10) # MHz
detune_freq_556 = 539.390 # THz

central_798 = 375.763240 # THz
delta_freq_list_798 = [0, 50, 100, 150, 200, 250, 300, 350] # MHz
detune_freq_798 = 375.767 # THz

if wavelength == '556':
    central = central_556
    delta_freq_list = delta_freq_list_556
    detune_freq = detune_freq_556
if wavelength == '798':
    central = central_798
    delta_freq_list = delta_freq_list_798
    detune_freq = detune_freq_798

print("central frequency set to:", central, "THz")
print("delta freq list:", delta_freq_list)
print("detune set to:", detune_freq, "THz")


"""connect to camera and set parameters"""
cam.connect(serial)
print("camera connected!")
cam.set_exposure(serial, exposure_time)
exposure = cam.get_exposure(serial)
print("exposure time set to:", exposure)
cam.set_gain(serial, gain)
print("gain set to:", gain)

def capture_images_pickle(cam, serial, num_shots, timeout_ms, save_dir):
    """获取pickle编码的图像并保存"""
    
    # 获取编码数据
    start_time = time.time()
    encoded = cam.fast_continuous_acquisition(serial, num_shots, timeout_ms)

    print("Acquisition complete. Decoding data...")
    
    if encoded.startswith("ERROR"):
        print(encoded)
        return []
    
    # 解码
    try:
        compressed = base64.b64decode(encoded.encode('ascii'))
        serialized = zlib.decompress(compressed)
        result = pickle.loads(serialized)
        
        images = result['images']
        count = result['count']
        shape = result['shape']
        
        print(f"Received {count} images, shape: {shape}")
        
        # 保存到本地
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving images to directory: {save_dir}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存为单个TIFF文件
            filename = os.path.join(save_dir, f"capture_{timestamp}.tiff")
            image_stack = np.array(images)
            tifffile.imwrite(filename, image_stack)
            
            print(f"Saved to: {filename}")
            
            # 也可以保存为多个文件
            for i, img in enumerate(images):
                single_filename = os.path.join(save_dir, f"image_{i:04d}.tiff")
                tifffile.imwrite(single_filename, img)
        end_time = time.time()
        print(f"Time taken to capture and save images: {end_time - start_time} seconds")
        return images
        
    except Exception as e:
        print(f"Decoding error: {e}")
        return []
    
def read_wl(freq, num):
    actual_freq_list = []
    error_list = []
    if wavelength == '556':
        channel = 1
    if wavelength == '399':
        channel = 3
    for i in range(num):
        actual_freq = ws8.read_wl_from_ws8(channel)
        error = np.abs(freq - actual_freq)
        actual_freq_list.append(actual_freq)
        error_list.append(error)
    max_error = np.max(error_list)
    avg_freq = np.mean(actual_freq_list)
    marker = True
    if max_error > 2e-6:  # 2 MHz tolerance
        marker = False
    return marker, avg_freq

def read_wl_channel_ntimes(target_freq, channel, ntimes):
    actual_freq_list = []
    error_list = []
    for i in range(ntimes):
        actual_freq = ws8.read_wl_from_ws8(channel)
        error = np.abs(target_freq - actual_freq)
        actual_freq_list.append(actual_freq)
        error_list.append(error)
    max_error = np.max(error_list)
    avg_freq = np.mean(actual_freq_list)
    marker = True
    if max_error > 2e-6:  # 2 MHz tolerance
        marker = False
    return marker, avg_freq
# def process_unit(cam, serial, freq, num_shots, time_out_ms, save_dir):
#     marker = False
#     start_time = time.time()
#     if freq == detune_freq:
#         ws8.unlock_lasers(['556'])
#         ws8.lock_lasers(['556'], [freq])
#         end_time = time.time()
#         print(f"Time taken for detune laser to far_detuning: {end_time - start_time} seconds")
#         capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
#     else:
#         while marker == False:
#             ws8.unlock_lasers(['556'])
#             print("556 unlocked")
#             marker0 = ws8.lock_lasers(['556'], [freq]) #lock the lasers
#             if marker0 == False:
#                 continue
#             marker1, avg_freq = read_wl(freq, 5)
#             marker = marker0 and marker1
#             print(f"avg_freq = {avg_freq}, marker0 = {marker0}, marker1 = {marker1}, marker = {marker}")
#         end_time = time.time()
#         print(f"Time taken for lock laser to {freq}: {end_time - start_time} seconds")
#         capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir) 
    

## old version for only 556 or 399 spetrum
def process_unit(cam, serial, freq, num_shots, time_out_ms, save_dir):
    marker = False
    start_time = time.time()
    
    # if freq == detune_freq:
    # flag = False
    if freq == detune_freq:
        ws8.unlock_lasers([wavelength])
        ws8.lock_lasers([wavelength], [freq])
        end_time = time.time()
        print(f"Time taken for detune laser to far_detuning: {end_time - start_time} seconds")
        capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
    else:
        while not marker:
            ws8.unlock_lasers([wavelength])
            print(f"{wavelength} unlocked")
            marker0 = ws8.lock_lasers([wavelength], [freq])
            marker0 = bool(marker0[0]) if isinstance(marker0, (list, np.ndarray)) else marker0
            if not marker0:
                continue
            # 内层循环，只要marker1为False就继续等待
            wait_time = 1
            wait_lock_time = time.time()
            while True:
                marker1, avg_freq = read_wl(freq, 20)
                # print(f"avg_freq = {avg_freq}, marker0 = {marker0}, marker1 = {marker1}")
                if marker1:
                    break
                # print(f"Frequency not stable, waiting {wait_time} seconds and retrying read_wl...")
                now_time = time.time()
                if now_time - wait_lock_time > 15:  # 超过15s 则直接unlock relock
                    ws8.unlock_lasers([wavelength])
                    ws8.lock_lasers([wavelength], [freq])
                time.sleep(wait_time)
            marker = marker0 and marker1
            print(f"avg_freq = {avg_freq}, marker0 = {marker0}, marker1 = {marker1}, marker = {marker}")
            end_time = time.time()
            print(f"Time taken for lock laser to {freq}: {end_time - start_time} seconds")
        capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
        

def lock_laser(laser_name, freq, far_detun = False):
    marker = False
    start_time = time.time()
    channel = 1
    wavelength = laser_name
    if laser_name == '556':
        channel = 1
    elif laser_name == '399':
        channel = 3
    else:
        print('Please check the name of laser')
        return -1
    if far_detun:
        ws8.unlock_lasers([wavelength])
        ws8.lock_lasers([wavelength], [freq])
        end_time = time.time()
        print(f"Time taken for lock {laser_name} to far_detuning at {freq}: {end_time - start_time} seconds")
    else:
        while not marker:
            ws8.unlock_lasers([wavelength])
            print(f"{wavelength} unlocked")
            marker0 = ws8.lock_lasers([wavelength], [freq])
            marker0 = bool(marker0[0]) if isinstance(marker0, (list, np.ndarray)) else marker0
            if not marker0:
                continue
            # 内层循环，只要marker1为False就继续等待
            wait_time = 1
            wait_lock_time = time.time()
            while True:
                marker1, avg_freq = read_wl_channel_ntimes(freq, channel, 10)
                if marker1:
                    break
                now_time = time.time()
                if now_time - wait_lock_time > 15:  # 超过15s 则直接unlock relock
                    ws8.unlock_lasers([wavelength])
                    ws8.lock_lasers([wavelength], [freq])
                time.sleep(wait_time)
            marker = marker0 and marker1
            print(f"avg_freq = {avg_freq}, marker0 = {marker0}, marker1 = {marker1}, marker = {marker}")
            end_time = time.time()
            print(f"Time taken for lock laser {laser_name} to {freq}: {end_time - start_time} seconds")

def experiment_process(cam, serial, central, delta_freq_list, detune_freq, num_shots, time_out_ms, path):
    for delta in delta_freq_list:
        print(f"delta = {delta}")

        save_dir = path + "/" + str(delta) + "/exp"
        freq = central + delta * 1e-6
        print(f"freq = {freq}, central_freq = {central}")
        process_unit(cam, serial, freq, num_shots, time_out_ms, save_dir)
        # time.sleep(10)
        time.sleep(0.1)

        save_dir = path + "/" + str(delta) + "/bg"
        freq = detune_freq
        print(f"freq = {freq}, central_freq = {central}")
        process_unit(cam, serial, freq, num_shots, time_out_ms, save_dir)
        # time.sleep(10)
        time.sleep(0.1)
    ws8.unlock_lasers([wavelength])
    print(f"{wavelength} unlocked at the end of experiment")

def lock_freq_continuous_img(cam, serial, freq, num_shots, time_out_ms, path, iter):
    """
    激光锁定在freq，连续拍照iter次，每次保存到path/第1次/拍照时间戳/exp
    只锁定一次激光
    """
    # 只锁一次激光
    ws8.lock_lasers([wavelength], [freq])
    start_time = time.time()
    print(f"Laser locked at {freq} THz")
    for i in range(iter):
        print(f"--- Continuous Iteration {i+1} of {iter} ---")
        save_dir = path+'/'+ '1' + '/'+ str(time.time()-start_time) +"/exp" ## save to same folder
        capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
        time.sleep(5)  # 可根据需要调整间隔
    print("Continuous image acquisition finished.")

## the following not finished yet, need to adjust
def lock_freq_continuous_img_with_bg(cam, serial, freq, detune_freq, num_shots, time_out_ms, path, iter):
    """
    激光锁定在freq，连续拍照iter次，每次保存到path/第1次/拍照时间戳/exp
    每次拍完exp后，调远失谐拍bg，保存到path/第1次/拍照时间戳/bg
    """
    start_time = time.time()
    print(f"Laser locked at {freq} THz")
    for i in range(iter):
        print(f"--- Continuous Iteration {i+1} of {iter} ---")
        lock_laser('556', freq, far_detun = False)
        time_stamp = time.time()-start_time
        save_dir_exp = path+'/'+ 'continuous' + '/'+ str(time_stamp) +"/exp" ## save to same folder
        capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir_exp)
        time.sleep(0.1)

        # 调远失谐拍bg
        ws8.unlock_lasers([wavelength])
        lock_laser('556', detune_freq, far_detun = False)
        save_dir_bg = path+'/'+ 'continuous' + '/'+ str(time_stamp) +"/bg" ## save to same folder
        capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir_bg)
        time.sleep(10)
    print("Continuous image acquisition with bg finished.")

def sweep_slow_beam_detuning_556img_locked(cam, serial, central_798, delta_freq_list_798, freq_556, far_detune_freq_556, num_shots, time_out_ms, path, iter):
    """
    扫描减速光频率
    556激光锁定在freq_556, 调远失谐拍背景。
    """
    for i in range(iter):
        for delta in delta_freq_list_798:
            print(f"798 detuning = {delta}, 399 detuningg = {2*delta}")
            freq = central_798 + delta * 1e-6
            lock_laser('399', freq, far_detun = False)

            save_dir = path + "/" + f"{i+1}" + "/" + str(delta) + "/exp"
            lock_laser('556', freq_556, far_detun=True)
            capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
            time.sleep(0.1)

            save_dir = path + "/" + f"{i+1}" + "/" + str(delta) + "/bg"
            lock_laser('556', far_detune_freq_556, far_detun = True)
            capture_images_pickle(cam, serial, num_shots, time_out_ms, save_dir)
            time.sleep(0.1)
            print(f"finished capture 556 fluor and bg imgs at 798 detuning {delta}, 399 detuning {2*delta}")
        ws8.unlock_lasers(['556'])
        ws8.unlock_lasers(['399'])
        print(f"556, 399 unlocked at the end of experiment")

def sweep_slow_beam_detuning_556img_spectrum(cam, serial, central_798, delta_freq_list_798, central_556, delta_freq_list_556, far_detune_freq_556, num_shots, time_out_ms, path, iter):
    for delta in delta_freq_list_798:
        print(f"798 detuning = {delta}, 399 detuningg = {2*delta}")
        freq_798 = central_798 + delta * 1e-6
        ws8.unlock_lasers(['399'])
        lock_laser('399', freq_798, far_detun = False)
        for i in range(iter):
            path = base_dir + "/" + folder_name + "/" + str(delta) + "/" + str(i+1)
            experiment_process(cam, serial, central_556, delta_freq_list_556, far_detune_freq_556, num_shots, time_out_ms, path)

        ws8.unlock_lasers(['556'])
        ws8.unlock_lasers(['399'])
        print(f"556, 399 unlocked at the end of experiment")


t0 = time.time()


# ws8.unlock_lasers(['399'])
# lock_laser('399', 375.763240-300e-6, far_detun = False)
# # ws8.lock_lasers(['399'], [375.763240-400e-6])  # pre-lock 399 laser to spectrum peak to slow atoms
# # # ws8.unlock_lasers(['556'])
# # # ws8.lock_lasers(['556'], [539.386589-0])  # pre-lock 399 laser to spectrum peak to slow atoms
for i in range(iter):
    print(f"=== Iteration {i+1} of {iter} ===")
    path = base_dir + "/" + folder_name + "/" + str(i+1)
    experiment_process(cam, serial, central, delta_freq_list, detune_freq, num_shots, time_out_ms, path)



## continuous img
# lock_freq_continuous_img_with_bg(cam, serial, central_556, detune_freq_556, num_shots, time_out_ms, os.path.join(base_dir, folder_name), iter = 20000)

## sweep 399 detuning and 556 locked
# sweep_slow_beam_detuning_556img_locked(cam, serial, central_798, delta_freq_list_798, central_556 - 50e-6, detune_freq_556, num_shots, time_out_ms, os.path.join(base_dir, folder_name), iter)

## sweep 399 detuning and 556 spectrum
# sweep_slow_beam_detuning_556img_spectrum(cam, serial, central_798, delta_freq_list_798, central_556, delta_freq_list_556, detune_freq_556, num_shots, time_out_ms, os.path.join(base_dir, folder_name), iter)

t1 = time.time()
print("time cost:", t1 - t0)



# sweep_slow_beam_detuning_556img_locked(cam, serial, central_399, delta_freq_list_399, central_556 - 100e-6, detune_freq_556, num_shots, time_out_ms, os.path.join(base_dir, folder_name), iter)


cam.disconnect(serial)


