import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

form = Tk()
form.title("Dự đoán phim")
form.geometry("500x400")

lable_ten = Label(form, text = "Nhập thông tin bộ phim:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_buying = Label(form, text = " Độ phổ biến:")
lable_buying.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_buying = Entry(form)
textbox_buying.grid(row = 2, column = 2)

lable_maint = Label(form, text = "Điểm IMDB:")
lable_maint.grid(row = 3, column = 1, pady = 10)
textbox_maint = Entry(form)
textbox_maint.grid(row = 3, column = 2)

lable_doors = Label(form, text = "Chi phí sản xuất:")
lable_doors.grid(row = 4, column = 1,pady = 10)
textbox_doors = Entry(form)
textbox_doors.grid(row = 4, column = 2)

lable_persons = Label(form, text = "Rotten Tomatoes:")
lable_persons.grid(row = 5, column = 1, pady = 10)
textbox_persons = Entry(form)
textbox_persons.grid(row = 5, column = 2)

lable_lug_boot = Label(form, text = "Tỷ suất lợi nhuận:")
lable_lug_boot.grid(row = 6, column = 1, pady = 10 )
textbox_lug_boot = Entry(form)
textbox_lug_boot.grid(row = 6, column = 2)

# Tạo nút bấm
button = tk.Button(form, text="Đánh giá")
button.grid(row=7, column=2, columnspan=1, rowspan=1, pady=10, padx=10, sticky="nswe")

form.mainloop()
