import os

path_to_json = '/cpfs/user/haoli84/code/InstanceDiffusion/dataset/daytimeclear_train_instdiff_data_withmask'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
with open('/cpfs/user/haoli84/code/InstanceDiffusion/dataset/daytimeclear_train_withmask.txt', 'w') as file:
    for l in json_files:
        file.write("{}/{}\n".format(path_to_json, l))
    file.close()
