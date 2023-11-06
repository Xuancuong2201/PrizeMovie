import tkinter as tk
from tkinter import *
from tkinter import messagebox

import numpy as np
from PIL import ImageTk, Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])


form = Tk()
form.title("Dự đoán phim")
form.geometry("720x400")
# Xây dựng GUI
lable_ten = Label(form, text="Nhập thông tin bộ phim:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row=1, column=1, padx=40, pady=10)

lable_popularity = Label(form, text="Độ phổ biến: (0 < Popularity < 1000)")
lable_popularity.grid(row=2, column=1, padx=0, pady=10)
textbox_popularity = Entry(form)
textbox_popularity.grid(row=2, column=2)

lable_IMDB = Label(form, text="Điểm IMDB: (0 < IMDB < 10)")
lable_IMDB.grid(row=3, column=1, pady=10)
textbox_IMDB = Entry(form)
textbox_IMDB.grid(row=3, column=2)

lable_Budget = Label(form, text="Chi phí sản xuất: (0 < Budget < 10)")
lable_Budget.grid(row=4, column=1, pady=10)
textbox_Budget = Entry(form)
textbox_Budget.grid(row=4, column=2)

lable_RT = Label(form, text="Rotten Tomatoes: (0 < RT < 100)")
lable_RT.grid(row=5, column=1, pady=10)
textbox_RT = Entry(form)
textbox_RT.grid(row=5, column=2)

lable_PM = Label(form, text="Tỷ suất lợi nhuận:")
lable_PM.grid(row=6, column=1, pady=10)
textbox_PM = Entry(form)
textbox_PM.grid(row=6, column=2)

# Xử lý dữ liệu
df = pd.read_csv("Movie.csv")
IQR = df.popularity.quantile(0.75) - df.popularity.quantile(0.25)
Lower_fence = df.popularity.quantile(0.25) - (IQR * 3)
Upper_fence = df.popularity.quantile(0.75) + (IQR * 3)
print("Popularity outliers are values < {lowerboundary} or > {upperboundary}".format(lowerboundary=Lower_fence,
                                                                                     upperboundary=Upper_fence))
IQR = df.budget.quantile(0.75) - df.budget.quantile(0.25)
Lower_fence = df.budget.quantile(0.25) - (IQR * 3)
Upper_fence = df.budget.quantile(0.75) + (IQR * 3)
print('Budget outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                 upperboundary=Upper_fence))
IQR = df.vote_average.quantile(0.75) - df.vote_average.quantile(0.25)
Lower_fence = df.vote_average.quantile(0.25) - (IQR * 3)
Upper_fence = df.vote_average.quantile(0.75) + (IQR * 3)
print('Vote average outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                       upperboundary=Upper_fence))
IQR = df.Tomatoes.quantile(0.75) - df.Tomatoes.quantile(0.25)
Lower_fence = df.Tomatoes.quantile(0.25) - (IQR * 3)
Upper_fence = df.Tomatoes.quantile(0.75) + (IQR * 3)
print('Tomatoes outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                   upperboundary=Upper_fence))
IQR = df.Profit_Margin.quantile(0.75) - df.Profit_Margin.quantile(0.25)
Lower_fence = df.Profit_Margin.quantile(0.25) - (IQR * 3)
Upper_fence = df.Profit_Margin.quantile(0.75) + (IQR * 3)
print('Profit Margin outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,
                                                                                        upperboundary=Upper_fence))
X = np.array(df[['popularity', 'budget', 'vote_average', 'Tomatoes', 'Profit_Margin']].values)
y = np.array(df['Success'])

# Chia X và y thành tập training và testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale dữ liệu
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train, y_train)
y_prectest = logreg.predict(X_test)

# SVM
model = SVC(kernel='linear', gamma=1)
model.fit(X_train, y_train)

print(model.coef_)
# Dự đoán theo Logictic Regression
y_Log = logreg.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của Logistic Regression: " + '\n'
                    + "Precision: " + str(precision_score(y_test, y_Log) * 100) + "%" + '\n'
                    + "Recall: " + str(recall_score(y_test, y_Log, average='macro') * 100) + "%" + '\n'
                    + "F1-score: " + str(f1_score(y_test, y_Log, average='macro') * 100) + "%" + '\n')
# Dự đoán theo SVM
y_SVM = model.predict(X_test)

lbl2 = Label(form)
lbl2.grid(column=3, row=8)
lbl2.configure(text="Tỉ lệ dự đoán đúng của Support Vector Machine: " + '\n'
                    + "Precision: " + str(accuracy_score(y_test, y_SVM) * 100) + "%" + '\n'
                    + "Recall: " + str(recall_score(y_test, y_SVM, average='macro') * 100) + "%" + '\n'
                    + "F1-score: " + str(f1_score(y_test, y_SVM, average='macro') * 100) + "%" + '\n')


def dudoanSVM():
    popurlarity = float(textbox_popularity.get())
    Profit_Margin = float(textbox_PM.get())
    Rottent_Tomotoes = float(textbox_RT.get())
    Budget = float(textbox_Budget.get())
    IMDB = float(textbox_IMDB.get())
    if ((popurlarity == '') or (Profit_Margin == '') or (Rottent_Tomotoes == '') or (Budget == '') or (IMDB == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([popurlarity, Budget, IMDB, Rottent_Tomotoes, Profit_Margin]).reshape(1, -1)
        X_dudoan = scaler.transform(X_dudoan)
        y_kqua = logreg.predict(X_dudoan)
        KQ = y_kqua
        open_new_window(KQ)
# Tạo nút bấm
button = tk.Button(form, text="Đánh giá", command=lambda: dudoanSVM())
button.grid(row=9, column=2, columnspan=1, rowspan=1, pady=10, padx=10, sticky="nswe")
button.config(bg="green", fg='white')

def open_new_window(KQ):
    new_window = tk.Toplevel()
    new_window.title("Kết quả đánh giá")
    new_window.geometry("300x300")

    if KQ == 1:
        image_path = "Success.png"
        text = "Đây là 1 bộ phim thành công"
    else:
        image_path = "Fail.png"
        text = "Đây là 1 bộ phim thất bại"
    image = Image.open(image_path)
    image = image.resize((50, 50))
    photo_image = ImageTk.PhotoImage(image)

    frame = tk.Frame(new_window)
    frame.pack()

    image_label = tk.Label(frame, image=photo_image)
    image_label.image = photo_image
    image_label.pack()
    text_label = tk.Label(frame, text=text)
    text_label.pack()


form.mainloop()
