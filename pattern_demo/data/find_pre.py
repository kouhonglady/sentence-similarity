# a = [[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]
# b = [[0.5,0.5],[0.4,0.6],[0.7,0.3]]
# pi = [0.2,0.4,0.4]
# o = [0,1,0]
# arr = []
# def pre():
#     n = len(a)
#     t = len(o)
#     for i in range(n):
#         temp =[0]*t
#         arr.append(temp)
#     for i in range(n):
#         arr[i][0] = pi[i] * b[i][o[0]]
#     print(arr)
#     for i in range(1,t,1):
#         for j in range(n):
#             temp = 0
#             for t in range(n):
#                 temp += arr[t][i-1]*a[t][j]
#             arr[j][i] = temp*b[j][o[i]]
#         print(arr)
#
#     result = 0
#     for i in range(n):
#         result += arr[i][n-1]
#     print(result)
#     print(max(arr[0]))
# # pre()
#
# arr1 = []
# def den():
#     n = len(a)
#     m = len(b[0])
#     t = len(o)
#
#     for i in range(n):
#         temp =[0]*t
#         arr1.append(temp)
#     for i in range(n):
#         arr1[i][t-1] = 1
#     print(arr1)
#     for i in range(n-2,-1,-1):
#         print(i)
#         for j in range(n):
#             for t in range(n):
#                 arr1[j][i] += arr1[t][i+1]*a[j][t]*b[t][o[i+1]]
#         print(arr1)
#
#
#     result = 0
#     for i in range(n):
#         result += arr1[i][0] * b[i][o[0]]*pi[i]
#     print(result)
#
#     print(t)
#     for i in range(t):
#         print(i)
#         tem = 0
#         for j in range(n):
#             tem += arr[j][i]*arr1[j][i]
#         print(tem)
#
# # den()
#
# def vtb():
#     n = len(a[0])
#     t = len(o)
#     x = []
#     y = []
#     for i in range(n):
#         temp = [0]*t
#         x.append(temp)
#     for i in range(n):
#         temp = [0]*t
#         y.append(temp)
#
#     for i in range(n):
#         x[i][0] = pi[i]*b[i][o[0]]
#         print(x[i][0])
#
#     for i in range(1,t,1):
#         for j in range(n):
#             temp = 0
#             pos = -1
#             for k in range(n):
#                 tt = x[k][i-1]*a[k][j]
#                 if tt > temp :
#                     temp = tt
#                     pos = k
#             x[j][i] = temp * b[j][o[i]]
#             y[j][i] = pos
#
#
#     result =0
#     position = -1
#     for i in range(n):
#         if x[i][t-1] > result :
#             result = x[i][t-1]
#             position = i
#     res = []
#     res.append(position)
#     print(position)
#     for i in range(t-2,-1,-1):
#         res.append(y[y[position][i+1]][i])
#         position = y[position][i+1]
#     print(res)
#     print(result)
#
# vtb()



#输入范例：
# singer_周杰|周杰伦|刘德华|王力宏;song_冰雨|北京欢迎你|七里香;actor_周杰伦|孙俪
# 请播放周杰伦的七里香给我听

# 输出范例:
# 请播放 周杰伦/actor,singer 的 七里香/song 给我听

# import sys
# def fine_end(word_dict, line, i):
#     for j in range(i,len(line))[::-1]:
#         if word_dict.get(line[i:j], [])!=[]:
#             return j
#     return -1
#
# if __name__ == '__main__':
#     # n = raw_input().strip().split(';')
#     n = 'singer_周杰|周杰伦|刘德华|王力宏;song_冰雨|北京欢迎你|七里香|你是风儿我是沙;actor_周杰伦|孙俪'.split(';')
#     word_dict = {}
#     put = ''
#     for i in n:
#         key = i.split('_')[0]
#         for j in i.split('_')[1].split('|'):
#             if word_dict.get(j, []) == []:
#                 word_dict[j] = [key]
#             else:
#                 word_dict[j].append(key)
#     #line = raw_input().strip()
#     line = '请播放周杰伦七里香和周杰的你是风儿我是沙给我听'
#     i = 0
#     print(word_dict)
#     while i < len(line):
#         e = fine_end(word_dict, line, i)
#         if e!=-1:
#             put += ' ' + line[i:e] + '/' +','.join(sorted(word_dict[line[i:e]])) + ' '#败在这个sorted上了，泪
#             i = e
#         else:
#             put += line[i]
#             i += 1
#     put = put.split()
#     put = ' '.join(put)
#     sys.stdout.write(put)


import pandas as pd

def main():
    dataset = pd.read_excel('E:/study/hrg_project/environment/dataset/test.xls',header = None)
    length = len(dataset)
    q1_result = []
    q2_result = []
    q2_original = []
    q2_total = []

    total = 0


    for i in range(length):
        sten1 = dataset.ix[i][2].strip('_').split("_")
        # print(sten1)
        q1_result.append(sten1[0])


    for i in range(length):
        sten1 = dataset.ix[i][3].strip('_').split("_")
        if q1_result[i] in sten1:
            q2_total.append(1)
            print(q1_result[i])
            total += 1
        else:
            q2_total.append(0)
        q2_original.append(sten1[0])

    count = 0
    count_original = 0
    print(len(q1_result))
    print(len(q2_result))
    for i in range(len(q2_result)):
        if q1_result[i] == q2_result[i]:
            count += 1
        # else:
        #     #print("%d : %s --- %s " % (count, q1_result[i], q2_result[i]))
    for i in range(len(q1_result)):
        if q1_result[i] == q2_original[i]:
            count_original += 1
        # else:
        #     # print("%d : %s --- %s " % (count_original, q1_result[i], q2_original[i]))

    total_count = 0
    for i in range(len(q2_total)):
        if q2_total == 1:
            total_count += 1


    print("the result is : %.10f"%(count/(len(q1_result)+ 0.1)))
    print("the result_original  is : %.10f" % (count_original / (len(q1_result) + 0.1)))
    print("the totoals is: %d ,and the rate is  :%.10f "%(total,total/len(q2_total)))

if __name__ == '__main__':
     main()


