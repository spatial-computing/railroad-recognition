import argparse

from datetime import datetime

import keras
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser(description='Test VGG model on test data')
parser.add_argument("img", help='path to raster map')
parser.add_argument("model", help='path to trained Keras model')
parser.add_argument("model_name", help='identifier of the test run')
parser.add_argument("--window_size", help='size, in pixels, of input data', default=47, type=int)

args = parser.parse_args()

img_path = args.img
model_path = args.model
model_name = args.model_name
window_size = args.window_size


# TODO: Make locations of ground truths files configurable
ground_truths_paths = {
    'buffer_0': '/datadrive/ssi/icdar/testing/data/testing_positive_coordinates_buffer0.txt',
    'buffer_1': '/datadrive/ssi/icdar/testing/data/testing_positive_coordinates_buffer1.txt',
    'buffer_2': '/datadrive/ssi/icdar/testing/data/testing_positive_coordinates_buffer2.txt',
    'buffer_3': '/datadrive/ssi/icdar/testing/data/testing_positive_coordinates_buffer3.txt'
}

ground_truths = {}


def load_ground_truths(gt_coords_path):
    gt_coords = []
    f = open(gt_coords_path, 'r')
    for line in f:
        gt_coords.append(line.rstrip('\r\n'))

    f.close()
    return frozenset(gt_coords)

# Load all coordinates for different files into ground_truths dictionary
for test_name, file_path in ground_truths_paths.iteritems():
    ground_truths[test_name] = load_ground_truths(file_path)


# Load model
model = keras.models.load_model(model_path)
assert model is not None

img = cv2.imread(img_path)
assert img is not None

img_height, img_width, img_channels = img.shape


class ModelBatchTester:
    """ Class that will generate predictions on an image test_img given a trained keras model.
        It will also [optionally] generate true/false positives/negatives if given a dictionary of ground_truths where
        a key (ground truth identifier) points to an array with shape (num_ground_truth_points, 2) that provides a list
        of (y, x) locations of pixels with positive labels.
    """

    def __init__(self, test_img, keras_model, model_name, ground_truths={}, step_size=1, window_size=47):
        self.test_img = test_img
        self.model = keras_model
        self.model_name = model_name
        self.step_size = step_size
        self.window_size = window_size
        self.ground_truths = ground_truths

        # Need to make sure window size is odd in order to have valid coordinates for the center
        assert window_size % 2 == 1

        self.batch_size = 1000000  # 1m

    def evaluate_prediction(self, prediction_coordinate, predicted_label, ground_truths_evaluations):
        """
        Evaluate prediction at a given point given point's prediction_coordinate (y,x), predicted_label, and an actual
        label in self.ground_truths

        :param prediction_coordinate: a 2-element array representing (y, x) coordinates of prediction
        :param predicted_label: label predicted by self.model for (y, x) coordinates
        :param ground_truths_evaluations: dictionary of ground_truth metrics, e.g.
            {'buffer0': {'tns': [[y1, x1], [y2,x2]], 'tps': [[y3, x3]], ... }, 'buffer1': {...} }

        """

        coord_str = ",".join(str(coord) for coord in prediction_coordinate)

        for test_name, evaluations in ground_truths_evaluations.iteritems():
            is_gt_positive = coord_str in self.ground_truths[test_name]

            if predicted_label == 0 and not is_gt_positive:
                evaluations.setdefault('tns', []).append(prediction_coordinate)
            elif predicted_label == 1 and not is_gt_positive:
                evaluations.setdefault('fps', []).append(prediction_coordinate)
            elif predicted_label == 0 and is_gt_positive:
                evaluations.setdefault('fns', []).append(prediction_coordinate)
            elif predicted_label == 1 and is_gt_positive:
                evaluations.setdefault('tps', []).append(prediction_coordinate)

    def predict_batch(self, test_data, centers, batch_enum):
        """
        Given an array the maximum size of self.batch_size and an array of center coordinates, apply self.model to
        generate prediction label for each point in :centers. Additionally, if self.ground_truths is not empty, it will
        use it to evaluate predicted label as true/false positive/negative.

        The method will then persist predictions and prediction evaluations (if present) to files that will have
        :batch_enum as suffix

        :param test_data: an array of images to predict on
        :param centers: an array of center coordinates (i-th coordinate in :centers represents location of the center of
                        i-th image in :test_data in the original self.test_img)

        :param batch_enum: current batch's number

        """
        positive_predictions = []

        predictions = self.model.predict(test_data, batch_size=128)
        predicted_labels = np.where(predictions[:,1] > 0.5, 1, 0)

        should_evaluate_predictions = len(self.ground_truths) != 0
        if should_evaluate_predictions:
            ground_truth_evaluations = {}
            for test_name in ground_truths.keys():
                evaluation_metrics = {
                    'fps': [],
                    'fns': [],
                    'tps': [],
                    'tns': []
                }

                ground_truth_evaluations[test_name] = evaluation_metrics

        for idx, predicted_label in enumerate(predicted_labels):
            center_coords = centers[idx]

            if predicted_label == 1:
                positive_predictions.append(center_coords)

            if should_evaluate_predictions:
                self.evaluate_prediction(center_coords, predicted_label, ground_truth_evaluations)

        # TODO: make output dir configurable
        output_dir = "/datadrive/ssi/icdar/testing/results/" + self.model_name + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.savetxt(output_dir + "ps_" + str(batch_enum).zfill(4), positive_predictions, fmt='%d')

        if should_evaluate_predictions:
            for test_name, evaluations in ground_truth_evaluations.iteritems():
                for evaluation_type, coords_array in evaluations.iteritems():
                    output_name = str(test_name) + "_" + str(evaluation_type) + "_" + str(batch_enum).zfill(4)
                    np.savetxt(output_dir + output_name, np.array(coords_array), fmt='%d')

    def process_batch(self, centers_batch, batch_enum):
        """
        For a list of center coordinates in centers_batch, generate a list of cropped test images according to
        self.window_size

        :param centers_batch: an array containing center coordinates (y, x)
        :param batch_enum: current batch's number
        """
        epoch_start = datetime.now()

        test = []
        centers = []

        for idx, center_coords in enumerate(centers_batch):
            center_y = center_coords[0]
            center_x = center_coords[1]

            min_y = center_y - (self.window_size - 1) / 2
            min_x = center_x - (self.window_size - 1) / 2
            max_y = min_y + self.window_size
            max_x = min_x + self.window_size
            test_datum = np.array(img[min_y:max_y, min_x:max_x]).astype(np.float32) / 255

            # if the resulting shape does not meet our model's input, ignore it
            if test_datum.shape[0] != window_size or test_datum.shape[1] != window_size:
                continue

            centers.append(center_coords)
            test.append(test_datum)

        assert len(test) == len(centers)

        self.predict_batch(np.array(test), np.array(centers), batch_enum)
        print "Finished batch " + str(batch_enum) + ": " + str((datetime.now() - epoch_start).total_seconds())

    def run_test(self):
        start_at = datetime.now()
        iterator = 0
        batch_enum = 0
        centers_batch = []
        epoch_start = datetime.now()

        # TODO: make bounding box coordinates configurable
        min_y = 741
        max_y = 14424
        min_x = 1350
        max_x = 11598

        windows_to_check = (max_y - min_y) * (max_x - min_x) / float((self.step_size **2))

        for y in xrange(min_y, max_y, self.step_size):
            for x in xrange(min_x, max_x, self.step_size):
                iterator += 1

                # Print out progress every 250k pixels
                if iterator % 250000 == 0:
                    print "completed " + str(iterator) + "/" + str(windows_to_check) + " windows in " + str(datetime.now() - epoch_start)
                    epoch_start = datetime.now()

                center_x = x + (window_size - 1) / 2
                center_y = y + (window_size - 1) / 2
                center = np.array([center_y, center_x])

                centers_batch.append(center)

                # We want to generate predictions in batches and make sure that there is enough memory to:
                # 1) store :centers_batch
                # 2) generate predictions for all elements in :centers_batch
                if iterator % self.batch_size == 0:
                    self.process_batch(centers_batch, batch_enum)

                    batch_enum += 1
                    centers_batch = []

        # Process last batch in case it's less than batch_size
        if len(centers_batch) > 0:
            self.process_batch(centers_batch, batch_enum)


tester = ModelBatchTester(img, model, model_name, ground_truths, window_size=window_size)
tester.run_test()
