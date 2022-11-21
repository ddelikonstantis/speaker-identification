from flask import Blueprint, render_template, request, redirect, flash, jsonify, make_response
from .process_audio import download_audio, create_segments, predict
from . import metadata


PATH_TO_CLIPS = 'downloads/parts/'
views = Blueprint('views', __name__)


def make_predict(file, segment_duration):
    # crops file into segment_duration-seconds segments
    segments = create_segments(path=file, segment_duration=segment_duration)
    pred_dict = predict(segments)

    return pred_dict


@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = None
        url = request.form.get('yb-url')
        upload = request.form.get('upload-id')
        if url:
            file = download_audio(url)
            print(f'Downloaded file: {file}')
        else:
            file = upload

        segment_duration = 2
        pred_dict = make_predict(file=file, segment_duration=segment_duration)
        return render_template("segments.html", title=file.rsplit('/', 1)[-1],  pred_dict=pred_dict, segment_duration=segment_duration)
    else:
        return render_template("home.html")


@views.route('/celebrities', methods=['GET'])
def celebrities():
    return render_template("celebrities.html", speakers=metadata['VGGFace1 ID'].to_dict())


@views.route('/upload', methods=['POST'])
def upload():
    if request.method == "POST":
        print(request.files)

        f = request.files['audio_data']
        file = 'src/visualization/downloads/' + f.filename + '.wav'
        with open(file, 'wb') as audio:
            f.save(audio)
        return jsonify(dict(redirect='segments', title=file))

    return render_template("home.html")
