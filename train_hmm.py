import numpy as np


# 对人工分词文件进行处理，提取出词和对应的词性[[(戴相龙,NR),(word,tag),(,)....],[],[]....]
def data_process(data_file):
    data_file = open(data_file, "r", encoding="utf-8")
    data_list = []
    sentence = []
    while True:
        line = data_file.readline()
        if not line:
            break
        if line != '\n':
            word = line.split()[1]
            tag = line.split()[3]
            sentence.append((word, tag))
        else:
            data_list.append(sentence)
            sentence = []
    data_file.close()
    return data_list


def creat_matrix(train_data):  # train_data存放所有的训练句子，[[(戴相龙,NR),(,),(,)....],[],[]....]
    tag_dict = {}  # tag_dict存放训练集中所有的tag，及其编号,考虑了起始和终止词性
    word_dict = {}  # word_dict存放训练集中所有的word，及其编号,加入了未知词
    for sentence in train_data:
        for word, tag in sentence:
            if word not in word_dict.keys():
                word_dict[word] = len(word_dict)
            if tag not in tag_dict.keys():
                tag_dict[tag] = len(tag_dict)

    tag_dict['BOS'] = len(tag_dict)
    tag_dict['EOS'] = len(tag_dict)
    word_dict['???'] = len(word_dict)
    # 将分词和词性以及对应编号以字典形式保存到文件中
    with open("word.dict", "w", encoding="utf-8") as word_dict_file:
        word_dict_file.write(str(word_dict))
    with open("tag.dict", "w", encoding="utf-8") as tag_dict_file:
        tag_dict_file.write(str(tag_dict))
    # 第(i,j)个元素表示词性j在词性i后面的概率（拓展了1行1列，最后一行是start，最后一列是stop）
    transition_matrix = np.zeros([len(tag_dict) - 1, len(tag_dict) - 1])
    # 第(i,j)个元素表示词性i发射到词j的概率
    emission_matrix = np.zeros([len(tag_dict) - 2, len(word_dict)])

    # 计算发射矩阵参数
    alpha = 0.1
    for sentence in train_data:
        for word, tag in sentence:
            emission_matrix[tag_dict[tag]][word_dict[word]] += 1
    for i in range(len(emission_matrix)):
        s = sum(emission_matrix[i])
        for j in range(len(emission_matrix[i])):
            emission_matrix[i][j] = (emission_matrix[i][j] + alpha) / (s + alpha * (len(word_dict)))  # 加alpha平滑

    # 计算转移矩阵参数
    for i in range(len(train_data)):
        for j in range(len(train_data[i]) + 1):
            if j == 0:
                transition_matrix[-1][tag_dict[train_data[i][j][1]]] += 1  # 初始tag频率
            elif j == len(train_data[i]):
                transition_matrix[tag_dict[train_data[i][j - 1][1]]][-1] += 1  # 结束tag频率
            else:
                transition_matrix[tag_dict[train_data[i][j - 1][1]]][tag_dict[train_data[i][j][1]]] += 1

    for i in range(len(transition_matrix)):
        s = sum(transition_matrix[i])
        for j in range(len(transition_matrix[i])):
            transition_matrix[i][j] = (transition_matrix[i][j] + alpha) / (s + alpha * (len(tag_dict) - 1))
    return transition_matrix, emission_matrix


def main():
    train_data = data_process('data/train.conll')
    transition_matrix, emission_matrix = creat_matrix(train_data)
    np.savetxt("transition_matrix.txt", transition_matrix)
    np.savetxt("emission_matrix.txt", emission_matrix)


main()
