import os
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
from tensorflow import keras
import math
import numpy as np

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'wav'}

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

model = keras.models.load_model("my_model")
labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def crop_music(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            #print("{}, segment:{}".format(file_path, d+1))

    return data


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

@app.route('/uploaded_file', methods=['GET', 'POST'])
def uploaded_file():
    file_path = os.listdir("upload")[0]
    os.rename("upload/" + file_path, "upload/test.wav")
    file_path = "upload/test.wav"
    new_data = crop_music(file_path, num_segments=10)
    mfcc = np.array(new_data["mfcc"])
    result = model.predict(mfcc)
    index = [0,0,0,0,0,0,0,0,0,0]
    for i in result:
        a = list(i)
        indx =a.index(max(a))
        index[indx] += 1
    prediction = index.index(max(index))
    os.remove(file_path)
    return labels[prediction]

if __name__ == "__main__":
    app.run(debug=True)
