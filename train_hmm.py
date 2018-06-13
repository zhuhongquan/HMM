import re
import collections


# 对文件数据进行处理，提取出词和对应的词性放到列表里
def data_processing(datafile):
    with open(datafile, "r", encoding="utf-8") as datafile:
        result_list = []
        for row in datafile.readlines():
            if row == "\n":
                continue
            else:
                word = re.findall(r'^\d+\s(.+?)\s_\s', row)
                tag = re.findall(r'_\s([A-Z]+?)\s_\s_\s\d+', row)
                temp_list = []
                temp_list.extend(word)
                temp_list.extend(tag)
                result_list.append(temp_list)
    return result_list


# 根据提取出的数据，建立状态转移矩阵
def creat_transition_dict(data_list):
    transition_dict = {}  # 用字典存放矩阵
    for i in range(len(data_list)-1):
        x = data_list[i][1]
        y = data_list[i+1][1]
        if x in transition_dict:
            if y in transition_dict[x]:
                transition_dict[x][y] += 1
            else:
                transition_dict[x][y] = 1
        else:
            transition_dict[x] = {}
            transition_dict[x][y] = 1
    state_list = transition_dict.keys()  # 获取所有的状态
    # 加1平滑并统计总数
    count_transition = 0
    for x in state_list:
        for y in state_list:
            if y not in transition_dict[x]:
                transition_dict[x][y] = 1
                count_transition += transition_dict[x][y]
            else:
                transition_dict[x][y] += 1
                count_transition += transition_dict[x][y]
    # 计算概率
    for x in state_list:
        for y in state_list:
            transition_dict[x][y] = transition_dict[x][y]/count_transition
    # 将转移矩阵以字典的形式保存到文件中
    with open("transition.dict", "w", encoding="utf-8") as transition_dict_file:
        transition_dict_file.write(str(transition_dict))

    '''
    # 打印状态转移矩阵
    print("      ", end='')
    for x in state_list:
        print("{:<9}".format(x), end='')
    print()
    for x in state_list:
        print("{:<6}".format(x), end='')
        for y in state_list:
            print("{:.6f}".format(transition_dict[x][y]) + ' ', end='')
        print()
    '''


# 计算发射(混淆)矩阵
def creat_emission_dict(data_list):
    emission_dict = {}
    data_tuple = []
    for i in data_list:
        data_tuple.append(tuple(i))  # 将列表转为元祖，为counter函数做准备
    dict1 = collections.Counter(data_tuple)  # 该字典记录的是混淆矩阵中对应着同一个分词的隐藏状态的数量
    # 统计每一个词性出现的频率(即隐藏状态出现的频率)
    state_list = []  # 存放词性(隐藏状态) 作为矩阵的行
    obs_list = []    # 存放分词(观测状态) 作为矩阵的列
    for i in data_list:  # 去除重复值
        if i[0] not in obs_list:
            obs_list.append(i[0])
        if i[1] not in state_list:
            state_list.append(i[1])

    for state in state_list:
        emission_dict[state] = {}  # 初始化混淆矩阵的隐藏状态
    # 统计数目并加1平滑
    for state in state_list:
        for obs in obs_list:  # 遍历所有观测状态
            temp_tuple = (obs, state)
            if temp_tuple in dict1:  # 查找(词,词性)出现的频率
                emission_dict[state][obs] = dict1[temp_tuple] + 1
            else:
                emission_dict[state][obs] = 1
    # 统计分母
    state_dict = {}
    for state in emission_dict:
        count_state = 0
        for obs in emission_dict[state]:
            count_state += emission_dict[state][obs]
        state_dict[state] = count_state
    # 计算概率
    for state in emission_dict:
        for obs in emission_dict[state]:
            emission_dict[state][obs] = emission_dict[state][obs]/state_dict[state]
    with open("emission.dict", "w", encoding="utf-8") as emission_dict_file:
        emission_dict_file.write(str(emission_dict))


# 计算初始概率
def get_start_probability(datafile):
    start_state = []
    datafile =  open(datafile, "r", encoding="utf-8")
    line = datafile.readline()
    while line != '':
        tag = re.findall(r'_\s([A-Z]+?)\s_\s_\s\d+', line)
        start_state.append(tag[0])
        while datafile.readline() != '\n':
            continue
        line = datafile.readline()
    start_prob = {}
    for tag in start_state:
        if tag not in start_prob:
            start_prob[tag] = 1
        else:
            start_prob[tag] += 1
    for tag in start_prob:
        start_prob[tag] = start_prob[tag] / len(start_state)
    with open("start_prob.dict", "w", encoding="utf-8") as start_prob_file:
        start_prob_file.write(str(dict(start_prob)))


def main():
    data_list = data_processing("train.conll")
    creat_transition_dict(data_list)
    creat_emission_dict(data_list)
    get_start_probability("train.conll")


main()
