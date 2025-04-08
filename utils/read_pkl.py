import pickle
import os

# 文件名
#file_name = '/home/ubuntu/project/Driving-with-LLMs/data/datanew_02.pkl' #their file
file_name = '/home/ubuntu/project/Driving-with-LLMs/data/process_data.pkl' #our file
#output_file_name = "first_row.pkl"  # 新文件名

# 检查文件是否存在
if not os.path.exists(file_name):
    print(f"文件 {file_name} 不存在。")
else:
        with open(file_name, 'rb') as file:
            data = pickle.load(file)

    # 获取第一行内容
first_row = data[1]
print(first_row)
'''
# 将第一行内容存储为一个新的pickle文件
with open(output_file_name, 'wb') as output_file:
    pickle.dump(first_row, output_file)
'''
#print(f"第一行内容已保存到文件 {output_file_name}。")