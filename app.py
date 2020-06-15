from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

clf=pickle.load(open('model86%.pkl','rb'))
df = pickle.load(open('processor.pkl', 'rb'))

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', df=df)


@app.route('/predict', methods=['POST'])
def predict():

    try:

        brand=request.form.get('brand')
        ram=request.form.get('ram')
        rom=request.form.get('rom')
        screen=request.form.get('screen')
        Primary_cam=request.form.get('Primary_cam')
        front_cam=request.form.get('front_cam')
        processor_cost=request.form.get('processor')
        screen=screen.lower().strip('inch')
        Primary_cam=Primary_cam.lower().strip('megapixel')
        front_cam=front_cam.lower().strip('megapixel')

        new = pd.DataFrame(columns=['screen_size', 'Primary_cam', 'front_cam', 'brand_Apple',
                                    'brand_Asus', 'brand_General', 'brand_Gionee', 'brand_Google',
                                    'brand_HTC', 'brand_Honor', 'brand_Huawei', 'brand_InFocus',
                                    'brand_LG', 'brand_Micromax', 'brand_Motorola', 'brand_Nokia',
                                    'brand_OnePlus', 'brand_Oppo', 'brand_Panasonic', 'brand_Realme',
                                    'brand_Samsung', 'brand_Vivo', 'brand_Xiaomi', 'ram_1', 'ram_2',
                                    'ram_3', 'ram_4', 'ram_6', 'ram_8', 'ram_12', 'rom_1', 'rom_8',
                                    'rom_16', 'rom_32', 'rom_64', 'rom_128', 'rom_256', 'rom_512'])
        d = {'brand': brand, 'screen_size': screen, 'Primary_cam': Primary_cam, 'front_cam': front_cam,
             'ram': ram, 'rom': rom}
        data = pd.DataFrame(d, index=[1, 2])
        F = pd.get_dummies(data, columns=['brand', 'ram', 'rom'])
        new[F.columns.values] = F[F.columns.values]
        new = new.replace(np.nan, 0)
        x = new.iloc[:, :].values
        p1=round(clf.predict(x)[0],2)
        p2=round(clf.predict(x)[0],2)/2.5
        pred=round(p1-p2)+int(processor_cost)

        return render_template('index.html', pred=pred, df=df, label=1)

    except:

        return render_template('index.html', label=0)


if __name__ =="__main__":
    app.run(debug=True)