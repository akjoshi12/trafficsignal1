# from flask import Flask , render_template , request
# from keras.applications import ResNet50 
# import numpy as np
# import pandas as pd
# import cv2
# from flask_cors import CORS


# app = Flask(__name__)
# CORS(app)

# resnet = ResNet50(weights= 'imagenet',input_shape=(224,224,3), pooling='avg')
# print("+"*50,"Model is loaded")

# labels = pd.read_csv("labels.txt").values
# @app.route('/',methods= ['POST'])
# def index():
#     return render_template("index.html", data = "Hello ")


# #@app.route('/example', method=['POST'])
# @app.route('/predict', methods= ['POST'])
# def predict():
#     img = request.files['image']
#     img.save("img.jpg")
#     image = cv2.imread("img.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (224,224))
#     image = np.reshape(image, (1,224,224,3))

#     pred = resnet.predict(image)
#     pred = np.argmax(pred)
#     pred = labels[pred]


#     return render_template("prediction.html" , data = pred)

# if __name__ == "__main__":
#     app.run(debug= True)

#######################################################################################################################################
##Running code

# from flask import Flask, render_template, request
# from keras.applications import ResNet50 
# import numpy as np
# import pandas as pd
# import cv2
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# resnet = ResNet50(weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
# print("+" * 50, "Model is loaded")

# labels = pd.read_csv("labels.txt").values

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     img = request.files['image']
#     img.save("img.jpg")
#     image = cv2.imread("img.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (224, 224))
#     image = np.reshape(image, (1, 224, 224, 3))

#     pred = resnet.predict(image)
#     pred = np.argmax(pred)
#     pred = labels[pred]

#     # Create HTML response instead of JSON
#     html_response = f"<h1>Prediction: {pred}</h1>"
#     return html_response

# if __name__ == "__main__":
#     app.run(debug=True)
##################################################################################################################################################

# #Merged Code
from flask import Flask, render_template, request, redirect, url_for
from keras.applications import ResNet50 
import numpy as np
import pandas as pd
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

resnet = ResNet50(weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
print("+" * 50, "Model is loaded")

labels = pd.read_csv("labels.txt").values

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        img = request.files['image']
        img.save("img.jpg")
        image = cv2.imread("img.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, (1, 224, 224, 3))

        pred = resnet.predict(image)
        pred = np.argmax(pred)
        pred = labels[pred]

        # Render the prediction result
        return render_template("index.html", data=pred, show_form=False)
    else:
        # Render the form
        return render_template("index.html", show_form=True)

if __name__ == "__main__":
    app.run(debug=True)








"""from flask import Flask, render_template, request, redirect, url_for
from keras.applications import ResNet50 
import numpy as np
import pandas as pd
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

resnet = ResNet50(weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
print("+" * 50, "Model is loaded")

labels = pd.read_csv("labels.txt").values

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        img = request.files['image']
        img.save("img.jpg")
        image = cv2.imread("img.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, (1, 224, 224, 3))

        pred = resnet.predict(image)
        pred = np.argmax(pred)
        pred = labels[pred]

        # Redirect to prediction page with prediction data
        return redirect(url_for('prediction', data=pred))
    else:
        # Render index.html
        return render_template("index.html")

@app.route('/prediction', methods = ['POST'])
def prediction():
    # Get the prediction data from the query parameter
    pred = request.args.get('data')

    # Render prediction.html with prediction data
    return render_template("prediction.html", data=pred)

if __name__ == "__main__":
    app.run(debug=True)"""




"""from flask import Flask, render_template, request, redirect
from keras.applications import ResNet50 
import numpy as np
import pandas as pd
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

resnet = ResNet50(weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
print("+" * 50, "Model is loaded")

labels = pd.read_csv("labels.txt").values

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect('/predict')
    return render_template("index.html", data="Hello ")

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img.save("img.jpg")
    image = cv2.imread("img.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.reshape(image, (1, 224, 224, 3))

    pred = resnet.predict(image)
    pred = np.argmax(pred)
    pred = labels[pred]

    # Create HTML response instead of JSON
    html_response = f"<h1>Prediction: {pred}</h1>"
    return html_response

if __name__ == "__main__":
    app.run(debug=True)"""

