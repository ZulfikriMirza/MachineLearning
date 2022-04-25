import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request
from app import app, db, bcrypt
from app.forms import SubmitForm, LoginForm, UpdateAccountForm
from app.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import neurolab as nl
import pickle
from LVQClassifier import LVQClassifier as LVQ
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


df = pd.read_csv('iris.csv')
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', tables=[df.to_html(classes='data', header="true")], titles=df.columns.values)


@app.route('/Prediksi', methods=['GET', 'POST'])
def prediksi():
    forms_input = SubmitForm()
    y_pred = ""
    if forms_input.validate_on_submit():
        a = forms_input.a.data
        b = forms_input.b.data
        c = forms_input.c.data
        d = forms_input.d.data
        x_test = np.array([a,b,c,d])
        x_test = x_test.reshape(1,4)
        print(x_test.shape)
        # test = x_test.reshape(1, -1)
        filename = 'app/Iris_asl_model.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(x_test)
        if y_pred == 1:
            y_pred = "bunga iris Setosa"
        elif y_pred == 2:
            y_pred = "bunga iris Versicolor"
        elif y_pred == 3:
            y_pred = "bunga iris Virginica"
    return render_template('Prediksi.html', forms_real=forms_input, y_pred= y_pred)

@app.route('/createmodel', methods=['GET', 'POST'])
def createmodel():
    #menampilkan dataset 'iris.csv' sebagai list pada output program
    dataset = pd.read_csv("iris.csv")
    #memasukkan nilai yang ada pada dataset ke x dan target ke y
    x = dataset.drop(["e"], axis=1).values
    y = dataset["e"].values
    #melihat data yang tersedia di setiap kerangka data untuk mengetahui berapa kategori yang ada,
    # dan membuat jumlah variabel dummy yang sesuai.
    dataf = pd.get_dummies(y)
    arr = [0 for x in range(61)]
    dataf['4'] = np.array(arr)
    target = dataf.values

    # Import train_test_split function
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Mengaktifkan/memanggil/membuat fungsi klasifikasi LVQ
    modellvq = LVQ()

    # Memasukkan data training pada fungsi klasifikasi LVQ
    lvqtrain = modellvq.fit(x_train, y_train)
    #menyimpan model dengan nama iris_asl_model.pkl untuk melakukan prediksi
    filename = 'Iris_asl_model.pkl'
    pickle.dump(lvqtrain, open(filename, 'wb'))
    print("Model created!!!")

    y_pred = lvqtrain.predict(x_test)
    print(x_test.shape)
    # Menentukan probabilitas hasil prediksi
    c = lvqtrain.predict_proba(x_test)
    print(c)
    # import confusion_matrix model
    confusion_matrix(y_test, y_pred)

    class1 = x[target[:, 0] > 0]
    class2 = x[target[:, 1] > 0]

    #memunculkan hasil plot
    plt.plot(confusion_matrix(y_test, y_pred))
    plt.plot(class1[:, 0], class1[:, 1], 'bo', class2[:, 0], class2[:, 1], 'go')
    plt.xlim(1, 8)
    plt.ylim(1, 8)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bunga Iris')
    print(classification_report(y_test, y_pred))
    #menyimpan hasil plot sebagai gambar agar nanti dimunculkan di website
    plt.savefig('static/grafik.png')

    return render_template('createmodel.html',  url = 'static/grafik.png')

@app.route("/about")
def about():
    return render_template('about.html', title='About')



@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)
