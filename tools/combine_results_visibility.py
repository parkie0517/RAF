import os
import re
import pandas as pd

def parse_and_format_final(folder_path):
    """
    지정된 폴더 경로를 탐색하여 데이터를 추출하고,
    BEV/3D를 별도 행으로, Condition 순서를 원본 파일에 맞춰 DataFrame으로 반환합니다.
    """
    long_format_results = []

    # 1. 데이터를 긴 형식(long format)으로 먼저 추출합니다.
    for root, dirs, files in os.walk(folder_path):
        if 'complete_results.txt' in files:
            file_path = os.path.join(root, 'complete_results.txt')
            try:
                # 파일 경로에서 epoch과 conf_threshold 추출
                path_parts = file_path.split(os.sep)
                epoch_folder = next(part for part in path_parts if part.startswith('epoch_'))
                epoch_match = re.search(r'epoch_(\d+)_total', epoch_folder)
                epoch = int(epoch_match.group(1))
                conf_threshold_index = path_parts.index(epoch_folder) + 1
                conf_threshold = float(path_parts[conf_threshold_index])

                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 각 Condition 블록을 파싱
                blocks = content.strip().split('\n\n')
                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) < 5: continue
                    condition = re.search(r'Condition: (.*)', lines[0]).group(1).strip()
                    bev_val = float(lines[3].split(':')[1].strip().split()[1])
                    d3_val = float(lines[4].split(':')[1].strip().split()[1])
                    
                    long_format_results.append({
                        'epoch': epoch, 'conf_threshold': conf_threshold,
                        'condition': condition, 'bev_0.5': bev_val, '3d_0.5': d3_val
                    })
            except (ValueError, IndexError, AttributeError) as e:
                print(f"파일 처리 중 오류가 발생하여 건너뜁니다: {file_path} - {e}")

    if not long_format_results:
        return pd.DataFrame()

    df_long = pd.DataFrame(long_format_results)
    
    # 2. 'bev_0.5'와 '3d_0.5' 컬럼을 'metric'과 'value'로 풀어줍니다 (Melt).
    df_melted = df_long.melt(
        id_vars=['epoch', 'conf_threshold', 'condition'],
        value_vars=['bev_0.5', '3d_0.5'],
        var_name='metric', value_name='value'
    )
    
    # --- Condition 순서 지정 부분 ---
    # 원하는 condition 순서를 리스트로 정의합니다.
    # condition_order = ['all', 'normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow', 'unnormal']
    condition_order = ['all', 'clean',  'partial', 'noisy' ]
    # 'condition' 컬럼을 위에서 정의한 순서를 따르는 카테고리형으로 변환합니다.
    df_melted['condition'] = pd.Categorical(df_melted['condition'], categories=condition_order, ordered=True)
    
    # 3. 풀어진 데이터를 다시 피벗하여 최종 형태로 만듭니다.
    df_final = df_melted.pivot_table(
        index=['epoch', 'conf_threshold', 'metric'],
        columns='condition',
        values='value'
    ).reset_index()
    
    # 4. metric 순서(BEV -> 3D)를 지정하고 최종 정렬합니다.
    df_final['metric'] = pd.Categorical(df_final['metric'], categories=['bev_0.5', '3d_0.5'], ordered=True)
    df_final = df_final.sort_values(by=['epoch', 'conf_threshold', 'metric']).reset_index(drop=True)
    
    return df_final


if __name__ == '__main__':
    model_name = 'eccv_supple' # name of the folder inside './logs/'
    # tag = 'test_crn_clr'
    # tag = 'test_robu_clr' 
    tag = 'test_samfusion' 
    
    folder_path = '/home/user/heejun/L4DR/logs/'+model_name+'/'+tag

    df_final_results = parse_and_format_final(folder_path)
    
    output_path = os.path.join(folder_path, f'results_{tag}.csv')
    df_final_results.to_csv(output_path, index=False)

    print(f"saved at: {output_path}")
