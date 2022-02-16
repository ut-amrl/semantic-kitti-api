import os

import csv
import argparse
import yaml
from auxiliary.laserscan import SemLaserScan
from auxiliary.laserscanpointsextractor_cluster import LaserScanPointsExtractorCluster, PointsForInstance

def getLabelsOfInterest():
    # car and moving-car
    # return [10, 252]
    return [10]

class PointClusterInstanceDetails:

    def __init__(self, points, sequence, scanNum, instNum, semanticClassNum):
        self.points = points
        self.sequence = sequence
        self.scanNum = scanNum
        self.instNum = instNum
        self.semanticClassNum = semanticClassNum

def getInstancesForSequence(datasetRoot, seqNumStr):

    # does sequence folder exist?
    scan_paths = os.path.join(datasetRoot, "sequences",
                              seqNumStr, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # does sequence folder exist?

    label_paths = os.path.join(FLAGS.dataset_root, "sequences",
                               seqNumStr, "labels")

    if os.path.isdir(label_paths):
        print("Labels folder exists! Using labels from %s" % label_paths)
    else:
        print("Labels folder doesn't exist! Exiting...")
        quit()
        # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check that there are same amount of labels and
    # if not FLAGS.ignore_safety:
    assert (len(label_names) == len(scan_names))

    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

    extractor = LaserScanPointsExtractorCluster(scan=scan,
                                         scan_names=scan_names,
                                         label_names=label_names, classes_of_interest=getLabelsOfInterest())

    points_by_scan_by_instance = extractor.getObjectInstancesForEachScan()
    instanceDetails = []
    for scanNum, pointsByInstance in points_by_scan_by_instance.items():
        for instNum, pointsDetails in pointsByInstance.items():
            instanceDetails.append(PointClusterInstanceDetails(pointsDetails.points_relative, seqNumStr, scanNum, pointsDetails.instance_num, pointsDetails.semantic_label))

    return instanceDetails

def argParser():
    parser = argparse.ArgumentParser("./points_with_instance_and_class_extractor.py")
    parser.add_argument(
        '--dataset_root', '-d',
        type=str,
        required=True,
        help='Directory to dataset root',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--output_dir', '-o',
        dest='output_dir',
        type=str,
        required=False,
        default="./output",
        help='Output %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        required=True,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    # parser.add_argument(
    #     '--scan_skip_num', '-n',
    #     type=int,
    #     required=False,
    #     default=40
    # )
    return parser.parse_known_args()

def generatePointsFileName(output_dir, sequence):
    baseFileName = "semantic_points_cars_seq_" + sequence + ".csv"
    seqOutFile = os.path.join(output_dir, baseFileName)
    return seqOutFile

def outputPointsForSeq(instancesForSeq, outputFileName):
    with open(outputFileName, 'w', newline='') as outputFile:
        csvWriter = csv.writer(outputFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["point_x", "point_y", "point_z", "semantic_label", "node_id", "cluster_label"])
        for instanceDetails in instancesForSeq:
            for rowIdx in range(instanceDetails.points.shape[0]):
               dataToWrite = [instanceDetails.points[rowIdx][0], instanceDetails.points[rowIdx][1], instanceDetails.points[rowIdx][2], instanceDetails.semanticClassNum, instanceDetails.scanNum, instanceDetails.instNum]
               csvWriter.writerow(dataToWrite)



if __name__ == "__main__":
    # centroidTest()
    FLAGS, unparsed = argParser()

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # TODO
    # Want to specify sequence id, number of frames to skip, classes of interest (can hard code these) and get out CSV file of points with the following fields
    # point_x, point_y, point_z, (all in lidar frame) semantic_label (as unsigned short), node id, cluster label

    instancesForSeq = getInstancesForSequence(FLAGS.dataset_root, FLAGS.sequence)
    filename = generatePointsFileName(FLAGS.output_dir, FLAGS.sequence)
    outputPointsForSeq(instancesForSeq, filename)



