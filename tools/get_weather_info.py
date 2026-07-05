# * run this file to check the weather of each sequence in the kradar dataset
 
import os

dataset_path = '/home/user/heejun/L4DR/data/kradar'

for i in range(1, 59):  # 1~58까지
    folder_path = os.path.join(dataset_path, str(i))
    desc_file = os.path.join(folder_path, 'description.txt')
    
    if os.path.exists(desc_file):
        with open(desc_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            if len(parts) >= 3:
                weather = parts[2].strip()
                print(f"{i}: {weather}")
            else:
                print(f"{i}: (날씨 정보 없음)")
    else:
        print(f"{i}: (description.txt 없음)")
