import numpy as np
import datetime
from train_HMM import data_process

# 读入分词和词性编号文件
print("正在读取参数文件...", end='')
word_dict_file = open("word.dict", "r", encoding="utf-8")
s1 = word_dict_file.read()
word_dict = eval(s1)
word_dict_file.close()
tag_dict_file = open("tag.dict", "r", encoding="utf-8")
s2 = tag_dict_file.read()
tag_dict = eval(s2)
tag_dict_file.close()
# 读入状态转移矩阵和发射矩阵
transition_matrix = np.loadtxt("transition_matrix.txt")
emission_matrix = np.loadtxt("emission_matrix.txt")
# 转换成对数
emission_matrix = np.log(emission_matrix)
transition_matrix = np.log(transition_matrix)
print("参数文件读取完毕")


def viterbi(word_list):
    word_index = []
    for word in word_list:
        if word in word_dict.keys():
            word_index.append(word_dict[word])
        else:
            word_index.append(word_dict['???'])

    observe = len(word_list) + 1  # 观测序列，即一个句子的分词序列
    states = len(tag_dict) - 2    # 隐藏状态，即分词可能的词性
    max_prop = np.zeros((observe, states))
    path = np.zeros((observe, states))

    for i in range(states):
        path[0][i] = -1
        max_prop[0][i] = transition_matrix[-1][i] + emission_matrix[i][word_index[0]]

    # 动态规划
    for i in range(1, observe):
        # 到达end状态有点区别
        if i == observe - 1:
            for k in range(states):
                max_prop[i][k] = max_prop[i - 1][k] + transition_matrix[k][-1]
                last_path = k
                path[i][k] = last_path
        else:
            for j in range(states):
                # 找出概率最大的路径(由于取了对数，所以采用加法)
                prob = max_prop[i - 1] + transition_matrix[:-1, j] + emission_matrix[j][word_index[i]]
                path[i][j] = np.argmax(prob)
                max_prop[i][j] = max(prob)

    gold_path = []
    cur_state = observe - 1
    step = np.argmax(max_prop[cur_state])
    while True:
        step = int(path[cur_state][step])
        if step == -1:
            break
        gold_path.insert(0, step)
        cur_state -= 1
    return gold_path


def evaluate(test_data):
    total_words = 0
    correct_words = 0
    sentence_num = 0
    print('正在标注词性...')
    f = open('predict.txt', 'w', encoding='utf-8')  # 预测结果保存在predict.txt中
    for sentence in test_data:
        sentence_num += 1
        word_list = []
        tag_list = []
        for word, tag in sentence:
            word_list.append(word)
            tag_list.append(tag)
        predict = viterbi(word_list)  # 调用viterbi函数解码
        total_words += len(sentence)
        for i in range(len(predict)):
            f.write(word_list[i] + '\t' + list(tag_dict.keys())[predict[i]] + '\n')
            if predict[i] == tag_dict[tag_list[i]]:
                correct_words += 1
        f.write('\n')
    f.close()
    print('共{}个句子'.format(sentence_num))
    print('共{}个单词，预测正确{}个分词'.format(total_words, correct_words))
    print('准确率：{}'.format(correct_words / total_words))


def main():
    test_data = data_process('data/dev.conll')
    start_time = datetime.datetime.now()
    evaluate(test_data)
    end_time = datetime.datetime.now()
    print('共耗时' + str(end_time - start_time))


main()
