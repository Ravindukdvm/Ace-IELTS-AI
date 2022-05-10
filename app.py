# Required Imports
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE";

import numpy as np
import streamlit as st
import easyocr

# Answer Keys
# For three different IELTS test types

# Academic Reading 
ar = {'01':['FALSE'], '02': ['FALSE'], '03': ['NOT GIVEN'], '04': ['TRUE'], '05': ['TRUE'], '06': ['FALSE'], '07':['TRUE'], '08': ['violent'], '09': ['tool'], '10': ['meat'], 
'11': ['photographer'], '12': ['game'], '13': ['frustration'], '14': ['IV', 'iv'], '15': ['VII', 'vii'], '16': ['II', 'ii'], '17': ['Vv', 'v', 'V'], '18': ['Ti', 'i'], '19': ['VIII', 'vii'], '20': ['VI', 'vi'], 
'21': ['city'], '22': ['priests'], '23': ['trench'], '24': ['location'], '25': ['DD'], '26': ['BB'], '27': ['BB'], '28': ['DD'], '29': ['CC'], '30': ['DD'], 
'31': ['GG'], '32': ['ee'], '33': ['CC'], '34': ['FF'], '35': ['BB'], '36': ['AA'], '37': ['CC'], '38': ['AA'], '39': ['BB'], '40': ['CC']}

# Academic Listening
al = {'01': ['Egg'], '02': ['Tower'], '03': ['Car'], '04': ['Animals'], '05': ['Bridge'], '06': ['Movie', 'film'], '07': ['Decorate'], '08': ['Wednesdays'], '09': ['Fradstone'], '10': ['Parking'], 
'11': ['CC'], '12': ['AA'], '13': ['BB'], '14': ['CC'], '15': ['HH'], '16': ['CC'], '17': ['GG'], '18': ['BB'], '19': ['II'], '20': ['AA'], 
'21': ['CC'], '22': ['EE'], '23': ['BB'], '24':['ee', 'EE'], '25': ['DD'], '26': ['cc', 'CC'], '27': ['AA'], '28': ['HH'], '29': ['FF'], '30': ['GG'], 
'31': ['practical'], '32': ['publication'], '33': ['choices'], '34': ['negative'], '35': ['play'], '36': ['capitalism'], '37': ['depression'], '38': ['logic'], '39': ['opportunity'], '40': ['Practice']}

# General Training Reading
gtr = {'01': ['DD'], '02': ['BB'], '03': ['CC'], '04': ['EE'], '05': ['AA'], '06': ['BB'], '07': ['TRUE'], '08': ['TRUE'], '09': ['NOT GIVEN'], '10': ['FALSE'], 
'11': ['FALSE'], '12': ['FALSE'], '13': ['TRUE'], '14': ['NOT GIVEN'], '15': ['fertilizer', 'fertiliser'], '16': ['animal'], '17': ['obstacle'], '18': ['aids'], '19': ['bending'], '20': ['gate'], 
'21': ['proactive'], '22': ['special offers'], '23': ['brand names'], '24': ['negativity'], '25': ['presentation'], '26': ['credit card'], '27': ['rudeness'], '28': ['Vi', 'VI', 'vi'], '29': ['iv', 'IV'], '30': ['ii', 'II'], 
'31': ['VIII', 'viii'], '32': ['Vv', 'V', 'v'], '33': ['Ti', 'Ii', 'i', 'I'], '34': ['iii', 'III'], '35': ['DD'], '36': ['DD'], '37': ['AA'], '38': ['family'], '39': ['platform'], '40': ['multi-coloured']}

# OCR Module
# Parameters: Input OpenCV Image
# Task:Extracts the text from the scanned document

def OCR(image):
    # Pre processing Steps: We can decide when to use and when not to  
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height = image.shape[0]
    width = image.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = image[:, :width_cutoff]
    s2 = image[:, width_cutoff:]

    # Converting to Grayscale   
    s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
    s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)

    # Fetching Text using EasyOCR   
    text1 = reader.readtext(s1, detail = 0)
    text2 = reader.readtext(s2, detail = 0)

    # Optional Spell Correcting Module   
    # text_new = [(lambda x: sp.correct(x))(x) for x in text]
    return text1 + text2



# Evaluation Module
# Parameters:
#     1. answers = Scanned Answers from uploaded sheet using OCR
#     2. type = IELTS Exam Type
# Task: Handles the evaluation of the scanned documents

def evaluate(answers, type="Academic Reading"):
    # Popping the name from scanned document
    answers.pop(0)
    score = 0
    name = answers.pop(0)

    # Initializing Answerkey Dictionary
    answerkey = dict()

    # Setting Answerkey according to Test Type
    if type == "Academic Reading":
        answerkey = ar
    elif type == "Academic Listening":
        answerkey = al
    elif type == "GT Reading":
        answerkey = gtr

    # Stored Incorrect Answers
    incorrect = []

    # Correcting the answers
    # Check if the key is there in answerkey, 
    # if it's there then check it with answer values and add score
    # or else just add to the incorrect answer
    for i in range(0, len(answers) - 1, 2):
        try:
            if answers[i] not in answerkey.keys():
                i+=1
            # print(f"i = {i}")
            if answers[i+1] in answerkey[answers[i]]:
                score+=1
            else:
                incorrect.append(answers[i+1])
        except KeyError:
            incorrect.append(answers[i+1])
            i+=1
            # continue
    print(f"Incorrect: {incorrect}")
    return score, name


# Streamlit Web Application module
if __name__ == '__main__':

    # Setting Page Configuration
    st.set_page_config(
     page_title="ACE THE IELTS",
     layout="centered",
     menu_items={
         'About': "##### This is an OCR based Automated Test Evaluation app" + 
         "which corrects the scanned answersheets and returns the score." +
         "This application supports three types of IELTS Test:"+ 
         "1. Academic Reading, 2. Academic Listening and 3. General Training Reading"
        }
    )

    # Setting Easy OCR Reader Model
    reader = easyocr.Reader(['en'], gpu=False)

    # Main App Body
    st.title("ACE THE IELTS")
    st.write(" ")

    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("OCR Model Used", "EasyOCR")
        col2.metric("Supported Languages", "80+")
        col3.metric("Supported IELTS Test Types", "3")

    image = cv2.imread(os.path.join(os.getcwd(), "images", "cover-1.jpg"))
    st.image(image, channels="BGR")

    # Paper Type Selector
    paper_type = st.selectbox("Choose IELTS Paper Type:", ["Academic Reading", "Academic Listening", "GT Reading"])

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], accept_multiple_files = False)

    # Evaluation Logic when file is uploaded
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        _, col, _ = st.columns(3)
        with col:
            st.image(opencv_image, channels="BGR")
            
            # Fetch text using OCR and do the Evaluation
            if st.button('Evaluate the Score'):
                summary = OCR(opencv_image)
                print(f"OCR Results: {summary}")
                score, name = evaluate(summary, paper_type)
                st.write(f"{name} scored {score}!")
                st.balloons()