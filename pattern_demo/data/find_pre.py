import pandas as pd

def main():
    dataset = pd.read_excel('E:/study/hrg_project/environment/dataset/test_new.xls',header = None)
    length = len(dataset)
    q1_result = []
    q2_result = []
    q2_original = []
    q2_total = []

    total = 0


    for i in range(length):
        sten1 = str(dataset.ix[i][2]).strip('_').split("_")
        # print(sten1)
        q1_result.append(sten1[0])


    for i in range(length):
        print(i)
        sten1 = dataset.ix[i][3].strip('_').split("_")
        if q1_result[i] in sten1:
            q2_total.append(1)

            total += 1
        else:
            q2_total.append(0)
            print(("%d:" + dataset.ix[i][1]) % i)
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


