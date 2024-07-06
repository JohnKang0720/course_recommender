from bs4 import BeautifulSoup
import pandas as pd
import requests

#PLAN: Course recommendation based on your profile

URL = 'https://vancouver.calendar.ubc.ca/course-descriptions/institution/120'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US, en;q=0.5",
}

# fetches all science subject links
def get_data(url):
    try:
        response=requests.get(url, HEADERS)
        soup=BeautifulSoup(response.text, 'html.parser')
    except:
        raise Exception("error!")
    return soup
    
# scrapes all course names, descriptions & more within each subject
def scrape_subjects():
    data = get_data(URL)
    links=data.find("ol", class_="list-buttons").find_all("a")
    n = len(links)
    
    course_df = pd.DataFrame()
    
    for i in range(9,10):
        temp_link = links[i]["href"]
        new_course = pd.DataFrame(scrape_each_subject(temp_link))
        course_df = pd.concat([course_df, new_course])
        # print(course_df)
    
    return course_df

# fetches all courses for each subject link
def scrape_each_subject(link):
    subject_courses = get_data(link).find("ol", class_="list-none").find_all("li")
    n=len(subject_courses)
    course_data=[]
    for j in range(0,n):
        course_name = subject_courses[j].find("strong").get_text()
        course_header = subject_courses[j].find("h3", class_="text-lg").get_text().replace(course_name, "").split("(")
        course_code = course_header[0]
        course_desc_all = subject_courses[j].find("p", class_="mt-0").get_text()
      
        course_data.append({
            "code": course_code,
            "name": course_name,
            "description": course_desc_all
        })
    
    return course_data

scrape_subjects().to_csv("./coursedata/course_data.csv")