from bs4 import BeautifulSoup
from selenium import webdriver
import re
import time
import xlrd
from xlutils.copy import copy
import os
import math
import pandas as pd



def spady():
    TOTAL_COUNT = 100000


    question_viewed = set()
    question_not_viewed = ["忘记手机银行登录密码怎么办"]
    pre_urls = "http://www.ccb.com/cn/home/customer_service/search-xyk.html?keyword="

    driver = webdriver.Chrome()
    total_count = 0

    filename = "data_from_network.xls"
    rb = xlrd.open_workbook(filename)
    wb = copy(rb)
    sheet = wb.get_sheet(0)

    while len(question_not_viewed) >0:
        print("test")
        if total_count > TOTAL_COUNT:
            break
        if (total_count+1)  %100 == 0:
            os.remove(filename)
            wb.save(filename)
            rb=xlrd.open_workbook(filename)
            wb = copy(rb)
            sheet = wb.get_sheet(0)
        question = question_not_viewed[0]
        question_not_viewed.remove(question)
        url = pre_urls + question
        question_viewed.add(question)
        print ("正在访问 :",url)
        driver.get(url)
        data = driver.page_source
        # print(data)
        soup = BeautifulSoup(data,'html.parser')
        title_list = soup.findAll("h2",{"class": "as-title"})
        content_list = soup.find_all("div", {"class": "as-content"})

        # print(len(title_list))
        # print(len(content_list))

        length = len(title_list)
        pattern_question = re.compile(r'<h2 class="as-title">|</h2>')
        pattern_answer = re.compile(r'<div class="as-content">|<p>|</p>|<br/>|<span class="as-time">|</span>|</div>')
        for i in range(length):
            string_question = re.sub(pattern_question, "", str(title_list[i]))
            string_answer = re.sub(pattern_answer, "", str(content_list[i]))
            if string_question not in question_viewed:
                question_not_viewed.append(string_question)
                sheet.write(total_count, 0, string_question)
                sheet.write(total_count, 1, string_answer)
                total_count += 1

        pattern = re.compile(r'<span class="cff0000">|</span>')
        print(soup.find("span",{"class": "cff0000"}))
        num_of_page = int(re.sub(pattern, "", str(soup.find("span",{"class": "cff0000"}))))

        page_number = int(math.ceil(num_of_page/7))
        print(page_number)


        for i in range(page_number):
            try:
                if i < page_number -1:
                    next = driver.find_element_by_class_name("nextPage")
                if next is not None:
                    next.click()
                    time.sleep(3)
                    data1 = driver.page_source
                    soup = BeautifulSoup(data1, 'html.parser')

                    title_list = soup.findAll("h2",{"class": "as-title"})
                    content_list = soup.find_all("div", {"class": "as-content"})

                    # print(len(title_list))
                    # print(len(content_list))

                    length = len(title_list)
                    pattern_question =re.compile(r'<h2 class="as-title">|</h2>')
                    pattern_answer = re.compile(r'<div class="as-content">|<p>|</p>|<br/>|<span class="as-time">|</span>|</div>')
                    for i in range(length):
                        string_question = re.sub(pattern_question, "",str(title_list[i]))
                        string_answer = re.sub(pattern_answer, "",str(content_list[i]))
                        if string_question not in question_viewed:
                            question_not_viewed.append(string_question)
                            sheet.write(total_count, 0, string_question)
                            sheet.write(total_count, 1, string_answer)
                            print(total_count)
                            total_count += 1
                            if (total_count + 1) % 100 == 0:
                                os.remove(filename)
                                wb.save(filename)
                                rb = xlrd.open_workbook(filename)
                                wb = copy(rb)
                                sheet = wb.get_sheet(0)
                            # print(string_question)
                            # print(string_answer)
            except:
                print("not find nextPage")
                wb.save(filename)


    driver.close()


def removedup():
    data = pd.read_excel(r"data_train.xlsx")
    print(data.shape)
    data.drop_duplicates(keep = False,inplace= True)
    print(data.shape)
    if os.path.isfile(r"data_train.xlsx"):
        os.remove(r"data_train.xlsx")  # 删除文件
    data.to_excel("data_train.xlsx",index=False, header=True)


def find_similary_question_on_baidu():
    data = pd.read_excel(r"test_new_pred_to_find_similarity.xlsx")
    print(data.shape)
    length = data.shape[0]

    pre_urls = "https://zhidao.baidu.com/search?ct=17&pn=0&tn=ikaslist&rn=10&fr=wwwt&word="
    driver = webdriver.Chrome()

    q1_q2_pairs = []

    for i in range(length):
        sten1 = data.ix[i ]['q1']
        urls = pre_urls + sten1
        print(i ," : " ,urls)
        driver.get(urls)
        page_source = driver.page_source
        # print(data)
        soup = BeautifulSoup(page_source, 'html.parser')
        best_ans_list = soup.findAll("dt", {"class": "dt mb-8"})
        top10_ans_list = soup.find_all("dt", {"class": "dt mb-4 line"})

        pattern = re.compile(r'title:[\u4e00-\u9fa5]*')
        pattern_zhongwen = re.compile(r'[^\u4e00-\u9fa5]')
        count = 0
        for item in top10_ans_list:
            result = pattern.findall(str(item))
            temp = re.sub(pattern_zhongwen,"",result[0])
            pair_temp = []
            pair_temp.append(sten1)
            pair_temp.append(temp)
            q1_q2_pairs.append(pair_temp)
            print(temp)
            count += 1
            if count == 4:
                break
        if i %200 == 0:
            if os.path.isfile(r"test_new_pred_to_find_similarity_result.xlsx"):
                os.remove(r"test_new_pred_to_find_similarity_result.xlsx")  # 删除文件
            result_data = pd.DataFrame(q1_q2_pairs)
            result_data.to_excel(r"test_new_pred_to_find_similarity_result.xlsx")

    print(len(q1_q2_pairs))
    result_data = pd.DataFrame(q1_q2_pairs)
    result_data.to_excel(r"test_new_pred_to_find_similarity_result.xlsx")

def dealdata():
    data = pd.read_excel(r"test_new_pred_to_find_similarity_result.xlsx")
    print()
    temp =data[data['q2'].str.len() > 5]
    print(temp.shape)
    if os.path.isfile(r"test_new_pred_to_find_similarity_result.xlsx"):
        os.remove(r"test_new_pred_to_find_similarity_result.xlsx")  # 删除文件
    temp.to_excel(r"test_new_pred_to_find_similarity_result.xlsx")

if __name__ == "__main__":
    # spady()

    # find_similary_question_on_baidu()
    # dealdata()
    # find_similary_question_on_baidu()
    # dealdata()
    removedup()