import re


def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]  # 列表里的元素为字典
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}

    # 初始化初始状态 (t == 0)
    start_states = []  # 隐藏状态(词性)集合
    for tag in start_p:
        start_states.append(tag)
    for y in start_states:
        if obs[0] not in emit_p['NR']:            # 对混淆矩阵中不存在的观察状态进行平滑处理
            V[0][y] = start_p[y] * 1/len(emit_p['NR'])
        else:
            V[0][y] = start_p[y] * emit_p[y][obs[0]]  # 通过字典关键字来查询概率
        path[y] = [y]

    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}  # 临时路径

        for y in states:
            if obs[t] not in emit_p['NR']:      # 对混淆矩阵中不存在的观察状态进行平滑处理
                (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * 1/len(emit_p['NR']), y0) for y0 in V[t-1])
            # 概率 隐状态 = 前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            else:
                (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in V[t-1])
            # 记录最大概率
            V[t][y] = prob
            # 记录路径
            newpath[y] = path[state] + [y]

        # 不需要保留旧路径
        path = newpath

    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  # 搜索出结尾概率最高的
    return (prob, path[state])


def main():
    start_prob_file = open("start_prob.dict", "r", encoding="utf-8")
    s1 = start_prob_file.read()
    start_probability = eval(s1)  # 初始状态概率
    start_prob_file.close()

    transition_file = open("transition.dict", "r", encoding="utf-8")
    s2 = transition_file.read()
    transition_probability = eval(s2)  # 状态转移矩阵

    states = []  # 隐藏状态(词性)集合
    for tag in transition_probability:
        states.append(tag)

    emission_file = open("emission.dict", "r", encoding="utf-8")
    s3 = emission_file.read()
    emission_probability = eval(s3)

    word_list = []
    with open('dev.conll', "r", encoding="utf-8") as inFile:
        temp_list = []  # 构造存放一整个句子分词的列表
        for s in inFile.readlines():
            if s == "\n":
                word_list.append(temp_list)
                temp_list = []  # 构造存放一整个句子分词的列表
            word = re.findall(r'^\d+\s(.+?)\s_\s', s)
            temp_list.extend(word)  # 将得到的分词存在句子临时列表里

    outfile = open("out.txt", "a", encoding="utf-8")
    for observations in word_list:
        (prob, best_path) = viterbi(observations, states, start_probability, transition_probability, emission_probability)
        n = 0
        for tag in best_path:
            outfile.write(observations[n]+'_'+tag+'\n')
            n += 1

main()