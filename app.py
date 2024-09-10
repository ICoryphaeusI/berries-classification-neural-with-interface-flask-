from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import autokeras as ak
import os

app = Flask(__name__)

TEMP_UPLOADS_FOLDER = 'temp_uploads'
app.config['TEMP_UPLOADS_FOLDER'] = TEMP_UPLOADS_FOLDER

# Загрузка лейбл-кодировщика
with open('91_version/label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Путь к модели
model_path = '91_version/best_autokeras_model'

# Словарь с обычными названиями классов
class_names_mapping = {
    'aronnik': 'Аронник пятнистый',
    'barbaris': 'Барбарис',
    'belladonna': 'Белладонна',
    'belokrilnik': 'Белокрыльник',
    'beresklet': 'Бересклет',
    'biryuchina': 'Бирючина',
    'boyaroshnik': 'Боярышник',
    'brusnika': 'Брусника',
    'buzina': 'Бузина',
    'buzina_yad': 'Бузина ядовитая',
    'cheremuha': 'Черемуха',
    'chernika': 'Черника',
    'ezevika': 'Ежевика',
    'fitolakka': 'Фитолакка',
    'gimolost': 'Жимолость',
    'gimolost_les': 'Жимолость лесная',
    'godgi': 'Годжи',
    'golubika': 'Голубика',
    'irga': 'Ирга',
    'kalina': 'Калина',
    'klukva': 'Клюква',
    'kostyanika': 'Костяника',
    'krizovnik': 'Крыжовник',
    'krushina': 'Крушина',
    'kupena': 'Купена',
    'landish': 'Ландыш',
    'magonia': 'Магония',
    'malina': 'Малина',
    'mogevelnik': 'Можжевельник',
    'moroshka': 'Морошка',
    'oblepiha': 'Облепиха',
    'paslen_chern': 'Паслен черный',
    'paslen_slad': 'Паслен сладкий',
    'ryabina': 'Рябина',
    'ryabina_cherno': 'Рябина черноплодная',
    'shelkovica': 'Шелковица',
    'shipovnik': 'Шиповник',
    'smorodina': 'Смородина',
    'snegno': 'Снежноягодник',
    'tis': 'Тис ягодный',
    'vinograd_dev': 'Виноград девичий',
    'vishnya': 'Вишня лесная',
    'volche_liko': 'Волчье лыко',
    'voronec': 'Воронец красноплодный',
    'voroniy_glaz': 'Вороний глаз',
    'zemlenika': 'Земляника'
}

# Словарь с соответствием русских и кодовых названий классов
russian_class_names_mapping = {
    'Аронник пятнистый': 'aronnik',
    'Барбарис': 'barbaris',
    'Белладонна': 'belladonna',
    'Белокрыльник': 'belokrilnik',
    'Бересклет': 'beresklet',
    'Бирючина': 'biryuchina',
    'Боярышник': 'boyaroshnik',
    'Брусника': 'brusnika',
    'Бузина': 'buzina',
    'Бузина ядовитая': 'buzina_yad',
    'Черемуха': 'cheremuha',
    'Черника': 'chernika',
    'Ежевика': 'ezevika',
    'Фитолакка': 'fitolakka',
    'Жимолость': 'gimolost',
    'Жимолость лесная': 'gimolost_les',
    'Годжи': 'godgi',
    'Голубика': 'golubika',
    'Ирга': 'irga',
    'Калина': 'kalina',
    'Клюква': 'klukva',
    'Костяника': 'kostyanika',
    'Крыжовник': 'krizovnik',
    'Крушина': 'krushina',
    'Купена': 'kupena',
    'Ландыш': 'landish',
    'Магония': 'magonia',
    'Малина': 'malina',
    'Можжевельник': 'mogevelnik',
    'Морошка': 'moroshka',
    'Облепиха': 'oblepiha',
    'Паслен черный': 'paslen_chern',
    'Паслен сладкий': 'paslen_slad',
    'Рябина': 'ryabina',
    'Рябина черноплодная': 'ryabina_cherno',
    'Шелковица': 'shelkovica',
    'Шиповник': 'shipovnik',
    'Смородина': 'smorodina',
    'Снежноягодник': 'snegno',
    'Тис ягодный': 'tis',
    'Виноград девичий': 'vinograd_dev',
    'Вишня лесная': 'vishnya',
    'Волчье лыко': 'volche_liko',
    'Воронец красноплодный': 'voronec',
    'Вороний глаз': 'voroniy_glaz',
    'Земляника': 'zemlenika'
}



@app.route('/')
def index():
    return render_template('index.html', russian_class_names_mapping=russian_class_names_mapping)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Создаем временную директорию, если ее нет
        if not os.path.exists(TEMP_UPLOADS_FOLDER):
            os.makedirs(TEMP_UPLOADS_FOLDER)

        # Сохраняем загруженный файл во временную директорию
        temp_filename = os.path.join(TEMP_UPLOADS_FOLDER, file.filename)
        file.save(temp_filename)

        # Загрузка модели внутри функции обработки запроса
        loaded_model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)

        # Обработка изображения
        new_img = cv2.imread(temp_filename, cv2.IMREAD_UNCHANGED)
        new_img = cv2.resize(new_img, (256, 256))
        new_img_array = np.expand_dims(new_img, axis=0)

        # Получение предсказанных вероятностей
        predictions = loaded_model.predict(new_img_array)

        # Преобразование вероятностей в исходные метки
        predicted_class = np.argmax(predictions)
        decoded_class = label_encoder.inverse_transform([predicted_class])[0]

        # Редирект на страницу предполагаемого класса с передачей пути к временному файлу
        return redirect(url_for('class_page', predicted_class=decoded_class, temp_filename=temp_filename))

@app.route('/class/<predicted_class>')
def class_page(predicted_class):
    temp_filename = request.args.get('temp_filename', '')

    # Здесь не нужно преобразовывать кодовое название в обычное
    class_name = predicted_class

    # Получим русское название из словаря
    russian_class_name = class_names_mapping.get(class_name, class_name)

    # Здесь вы можете передать информацию о ягоде на страницу с классом
    # Используйте russian_class_name на странице для отображения русского названия
    return render_template(f'{class_name}.html', predicted_class=russian_class_name, temp_filename=temp_filename, russian_class_names_mapping=russian_class_names_mapping)



if __name__ == '__main__':
    app.run(debug=True)
