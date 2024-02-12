import silence_tensorflow.auto
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth


global model

def importdata(): 
    balance_data = pd.read_csv('dataset/dataset.txt')
    balance_data = balance_data.abs()
    rows = balance_data.shape[0]  # gives number of row count
    cols = balance_data.shape[1]  # gives number of col count
    return balance_data


def splitdataset(balance_data):
    X = balance_data.values[:, 0:8] 
    y_ = balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    print(Y)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    return train_x, test_x, train_y, test_y


def GenerateModel(request):
    global model
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(2, activation='softmax', name='output'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('CNN Neural Network Model Summary: ')
    print(model.summary())
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100
    context = 'ANN Accuracy : ' +str(ann_acc)
    if request == "GET":
        return context
    else:
        if request == "POST":
            return model


def UserCheck(request, model):
    input = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n';
    input += request+"\n"
    f = open("dataset/test.txt", "w")
    f.write(input)
    f.close()
    test = pd.read_csv('dataset/test.txt')
    test = test.values[:, 0:8]
    # predict = model.predict_classes(test)
    predict = np.argmax(model.predict(test), axis=-1)
    print(predict[0])
    msg = ''
    if str(predict[0]) == '0':
        msg = "Given Account Details Predicted As Genuine"
    if str(predict[0]) == '1':
        msg = "Given Account Details Predicted As Fake"
    context = msg
    return context


def ViewTrain():
    data = pd.read_csv('dataset/dataset.txt')
    return data


st.title("Spot The Fake App") 

image_file_path = "images/f4.jpg"

# heading = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
#           ' font-size: 20px;">Developed by Vijay</p'
# st.write(heading, unsafe_allow_html=True)

# st.write( "<font color = ‘red’>THIS TEXT WILL BE RED</font>", unsafe_allow_html=True)

add_dropbox = st.sidebar.selectbox(
    "Choose Input Source",
    ("Home", "Admin", "User Check")
)

if add_dropbox == "Home":
    image = np.array(Image.open(image_file_path))
    st.image(image)
    st.write("In this project, we use machine learning, namely an \
    artificial neural network to determine what are the chances that \
    Facebook friend request is authentic or not. We also outline the \
    classes and libraries involved.")

    st.write("Furthermore, we discuss the relu \
    function and how the weights are determined and used. Finally, we \
    consider the parameters of the social network page which are utmost \
    important in the provided solution.")

elif add_dropbox == "Admin":
    names = ['Vijay', 'Admin']
    usernames = ['vijay', 'admin']
    passwords = ['vijay', 'admin']

    hashed_passwords = stauth.hasher(passwords).generate()

    authenticator = stauth.authenticate(names,usernames,hashed_passwords,
    'app_cookie', '#123', cookie_expiry_days=0)

    name, authentication_status = authenticator.login('Login')

    if authentication_status:
        st.write('Welcome *%s*' % (name))
        Filters = st.sidebar.radio("Choose among given operations:",
                            ("Generate Model", "View Train Data")
                            )
        if Filters == "Generate Model":
            image = np.array(Image.open(image_file_path))
            st.image(image)
            st.write("After clicking on the generate button,\
                    Please wait for the ANN model to be generated and for the accuracy to be displayed.")
            if st.button("Geneate"):
                st.write(GenerateModel("GET"))

        elif Filters == "View Train Data":
            text = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
            ' font-size: 20px;">Training data in the form of DataFrame:</p'
            st.write(text, unsafe_allow_html=True)
            st.dataframe(ViewTrain())
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


elif add_dropbox == "User Check":
    image = np.array(Image.open(image_file_path))
    st.image(image)
    input1 = st.text_input("Enter Profile Details in Numericals: ")
    if st.button("Check"):
        st.write(UserCheck(input1, GenerateModel("POST")))
    warning = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
          ' font-size: 20px;">Please wait for the Ann model to display final message!:</p'
    st.write(warning, unsafe_allow_html=True)

    instructions = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
          ' font-size: 20px;">User Instructions:</p'
    st.write(instructions, unsafe_allow_html=True)
    st.write("* Press Enter after providing the input as mentioned below and click the button.")
    st.write("* Enter eight input values in the format given below:")
    st.write("* Account_Age, Gender, User_Age, Link_Desc, Status_Count, Friend_Count, Location, Location_IP")
    st.write("* Enter value as '0', if the respective Gender, Link_Desc, Location, Location_IP exist on profile.")
    st.write("* Enter value as '1', if the respective Gender, Link_Desc, Location, Location_IP does not exist on profile.")
    st.write("* Please refer the sample input details of a genuine profile: 12,0,30,0,10962,958,0,0")
    # st.write("* Start spotting the fakes!")
