import os

import numpy as np
from numpy.linalg import inv
import argparse
import yaml
from auxiliary.laserscan import SemLaserScan
from auxiliary.laserscanpointsextractor_cluster import LaserScanPointsExtractorCluster, PointsForInstance

#
# things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
# stuff = [
#     'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
#     'traffic-sign'
# ]
def getLabelsOfInterest():
    # Not entirely clear which classes are 'stuff' and which are 'things'
    # Being conservative for now
    # 50 and 51 might not be... want to check if there are multiple instances for these across a sequence
    # return [10, 11, 13, 15, 18, 20, 30, 31, 32, 50, 99, 252, 253, 254, 255, 256, 257, 258]
    return [10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259]


class InstanceDetails:

    def __init__(self, points, globalPose, sequence, scanNum, instNum, semanticClassNum):
        self.points = points
        self.globalPose = globalPose
        self.sequence = sequence
        self.scanNum = scanNum
        self.instNum = instNum
        self.semanticClassNum = semanticClassNum

def getInstanceNumsForSemanticClass(instanceDetailsList):
    instanceNumsBySemanticClass = {}
    for instanceDetails in instanceDetailsList:
        instancesForClass = instanceNumsBySemanticClass.get(instanceDetails.semanticClassNum, [])
        instancesForClass.append(instanceDetails.instNum)
        instanceNumsBySemanticClass[instanceDetails.semanticClassNum] = instancesForClass
    return {semanticClass:set(instNums) for semanticClass, instNums in instanceNumsBySemanticClass.items()}

def getCentroidAndNormalizedPoints(pointsForInst):
    # points for inst is numpy array of x y z reflectance
    meanVal = pointsForInst.mean(axis=0)
    meanVal[3] = 0
    normalizedPoints = pointsForInst - meanVal
    return (meanVal[0:3], normalizedPoints)

def transformPointsForInstance(pointsStruct, pose):
    # print(pointsStruct.points_relative)
    reflectance = pointsStruct.points_relative[:, 3]
    onesColumn = np.ones((pointsStruct.points_relative.shape[0], 1))
    homogeneousPoints = np.hstack((pointsStruct.points_relative[:, 0:3], onesColumn))
    # print(homogeneousPoints)
    # print(pose.shape)
    # print(pose)
    transformedPoints = np.transpose(np.matmul(pose, np.transpose(homogeneousPoints)))
    # print(reflectance.shape)

    remissions_2d = np.reshape(reflectance, (reflectance.shape[0], 1))
    # print(transformedPoints)
    transformedPointsWithRemissions = np.concatenate((transformedPoints[:, 0:3], remissions_2d), axis=1)
    # print(transformedPointsWithRemissions)

    return PointsForInstance(pointsStruct.scan_num, pointsStruct.instance_num, transformedPointsWithRemissions, pointsStruct.semantic_label)

def transformPointsSingleScan(pose, points_by_instance):
    return {instance:transformPointsForInstance(pointsStruct, pose) for (instance, pointsStruct) in points_by_instance.items()}

def transformPoints(points_by_scan_by_instance, poses):
    return {scan_num:transformPointsSingleScan(poses[scan_num], points_by_inst) for (scan_num,points_by_inst) in points_by_scan_by_instance.items()}


def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  print(Tr)
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    #TODO Why the pre multiply by Tr_inv?
    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    # poses.append( np.matmul(pose, Tr))

  return poses

def getInstancesForSequence(datasetRoot, seqNumStr):

    calibration = parse_calibration(os.path.join(FLAGS.dataset_root, "sequences", seqNumStr, "calib.txt"))
    # suma_poses = parse_poses(os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "poses.txt"), calibration)
    poses = parse_poses(os.path.join(FLAGS.dataset_root, "poses", seqNumStr + ".txt"), calibration)

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
    # print(points_by_scan_by_instance)

    transformed = transformPoints(points_by_scan_by_instance, poses)

    instanceDetails = []
    for scanNum, pointsByInstance in transformed.items():
        for instNum, pointsDetails in pointsByInstance.items():
            unnormalizedRemissions = pointsDetails.points_relative[:, 3]
            (globalPose, normalizedPoints) = getCentroidAndNormalizedPoints(pointsDetails.points_relative)
            remissions = normalizedPoints[:, 3]
            if (not np.array_equal(unnormalizedRemissions, remissions)):
                print("Normalization changed the remissions")
            if (np.amax(remissions) > 1):
                print("Remission outside of range: " + str(np.amax(remissions)))
            if (np.amin(remissions) < 0):
                print("Remission below min range: " + str(np.amin(remissions)))

            instanceDetails.append(InstanceDetails(normalizedPoints, globalPose, seqNumStr, scanNum, pointsDetails.instance_num, pointsDetails.semantic_label))

    # print(instanceDetails)
    # instNumsBySemanticClass = getInstanceNumsForSemanticClass(instanceDetails)
    # print("Inst nums for semantic class")
    # for semanticClass, instNums in instNumsBySemanticClass.items():
    #     print("Class: " + str(semanticClass) + ", Instances: " + str(instNums))
    return instanceDetails



def generateFileNameWithSuffix(outDir, sequence, scanNum, instNum, semClass, suffix):
    baseFileName = "sem_kitti_cluster_" + sequence + "_scan" + str(scanNum) + "_inst" + str(instNum) + "_semClass" + str(semClass) + "_" + suffix + ".npy"
    seqOutDir = os.path.join(outDir, sequence)
    if not os.path.exists(seqOutDir):
        os.makedirs(seqOutDir)
    return os.path.join(seqOutDir, baseFileName)

def generateGlobalPoseFileName(outDir, sequence, scanNum, instNum, semClass):
    return generateFileNameWithSuffix(outDir, sequence, scanNum, instNum, semClass, "globPose")

def generatePointsFileName(outDir, sequence, scanNum, instNum, semClass):
    return generateFileNameWithSuffix(outDir, sequence, scanNum, instNum, semClass, "points")

def outputGlobalPose(fileName, globalPose):
    np.save(fileName, globalPose)

def outputPoints(fileName, points):
    np.save(fileName, points)

def outputInstanceData(instance, outDir):
    globalPoseFileName = generateGlobalPoseFileName(outDir, instance.sequence, instance.scanNum, instance.instNum, instance.semanticClassNum)
    pointsFileName = generatePointsFileName(outDir, instance.sequence, instance.scanNum, instance.instNum,
                                                    instance.semanticClassNum)
    outputGlobalPose(globalPoseFileName, instance.globalPose)
    outputPoints(pointsFileName, instance.points)


def outputInstances(instances, outDir):
    for instance in instances:
        outputInstanceData(instance, outDir)

def extractInstances(datasetRoot):
    sequenceNums = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    # sequenceNums = ["01"]
    instances = []
    for seqNum in sequenceNums:
        instancesForSeq = getInstancesForSequence(datasetRoot, seqNum)
        # instances.extend(getInstancesForSequence(datasetRoot, seqNum))
        outputInstances(instancesForSeq, FLAGS.output_dir)
        instances.extend(instancesForSeq)
    return instances

def centroidTest():
    mat = np.arange(20).reshape(5, 4)
    mat[0, :] = mat[0, :] - 1
    print(mat)
    print(getCentroidAndNormalizedPoints(mat))


def argParser():
    parser = argparse.ArgumentParser("./instance_points_extractor.py")
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
    return parser.parse_known_args()

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

    instances = extractInstances(FLAGS.dataset_root)
