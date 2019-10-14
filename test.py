"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from flask import Flask, jsonify, request, redirect
from flask import jsonify
import shutil
from flask_cors import CORS, cross_origin

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
train_directory = ''
test_directory = ''

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_PATH'] = './client'
model_dir = ''
person = ''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/train', methods=['POST'])
@cross_origin()
def get_samples_to_train():
    # Check if a valid image file was uploaded
    if request.method == 'POST':

        if 'samples' not in request.files:
            print('Samples not in the uploaded files')
            return redirect(request.url)

        samples = request.files.getlist("samples")
        print(samples)

        client_name = get_client_name()
        client_directory = get_client_dir()
        train_directory = get_training_dir()
        model_dir = get_model_dir()

        if 'person' not in request.headers:
            return redirect(request.url)

        person = request.headers['person']

        if not os.path.exists(train_directory):
            os.makedirs(train_directory)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # if samples.filename == '':
        #     return redirect(request.url)

        # if file and allowed_file(file.filename):
        # STEP 1: Train the KNN classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        # print("Training KNN classifier...")
        # classifier = train("train_dir", model_save_path="trained_knn_model.clf", n_neighbors=2)
        # print("Training complete!")
        if not os.path.exists(train_directory + os.path.sep + person):
            os.makedirs(train_directory + os.path.sep + person)

        for f in request.files.getlist('samples'):
            f.save(os.path.join(train_directory + os.path.sep + person, f.filename))

        # if file and allowed_file(file.filename):
        # STEP 1: Train the KNN classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        # print("Training KNN classifier...")
        classifier = train(train_directory, model_save_path=model_dir + os.path.sep + "trained_knn_model.clf",
                           n_neighbors=2)
        print("Training complete!")

        return {'status': 'Training Complete!'}
    else:
        return {'status': 'This method is not allowed!'}


@app.route('/predict', methods=['POST'])
@cross_origin()
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        test_directory = get_test_dir()

        if not os.path.exists(test_directory):
            os.makedirs(test_directory)
        else:
            delete_folder(test_directory)
            os.makedirs(test_directory)  # Start with an empty test directory

        for f in request.files.getlist('file'):
            f.save(os.path.join(get_test_dir(), f.filename))

        if file and allowed_file(file.filename):
            # STEP 1: Train the KNN classifier and save it to disk
            # Once the model is trained and saved, you can skip this step next time.
            # print("Training KNN classifier...")
            # classifier = train("train_dir", model_save_path="trained_knn_model.clf", n_neighbors=2)
            # print("Training complete!")
            results = []

            # The image file seems valid! Detect faces and return the result.
            for image_file in os.listdir(test_directory):
                full_file_path = os.path.join(test_directory, image_file)

                print("Looking for faces in {}".format(image_file))

                # Find all people in the image using a trained classifier model
                # Note: You can pass in either a classifier file name or a classifier model instance
                predictions = predict(full_file_path,
                                      model_path=get_model_dir() + os.path.sep + "trained_knn_model.clf")

                sub_results = []
                # Print results on the console
                for name, (top, right, bottom, left) in predictions:
                    sub_results.append( { "person": name,
                                    "coordinates": {"left": left, "top": top}
                                    })
                    # results.append("- Found {} at ({}, {})".format(name, left, top))
                    # print("- Found {} at ({}, {})".format(name, left, top))

                results.append({
                    "filename": image_file,
                    "faces": sub_results
                })
            return jsonify(results)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def delete_folder(path):
    shutil.rmtree(path)


def get_model_dir():
    return get_client_dir() + os.path.sep + 'model'


def get_client_name():
    if 'client' not in request.headers:
        return 'none'
    else:
        return request.headers['client']


def get_client_dir():
    return app.config['UPLOAD_PATH'] + os.path.sep + get_client_name()


def get_training_dir():
    return get_client_dir() + os.path.sep + 'train_directory'


def get_test_dir():
    return get_client_dir() + os.path.sep + 'test_directory'


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
