#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanpointsextractor import LaserScanPointsExtractor, PointsForInstance
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
# import open3d.geometry as geom
import open3d
import csv
import math
import joblib

class PointsForInstAllScans:

    def __init__(self, instance_num, points_relative, semantic_label):
        # self.scan_num = scan_num
        self.instance_num = instance_num
        self.points_relative = points_relative
        self.semantic_label = semantic_label

class Centroid:

    def __init__(self, instance_num, centroid_point, semantic_label):
        self.instance_num = instance_num
        self.centroid_point = centroid_point
        self.semantic_label = semantic_label



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

def transformPointsForInstance(pointsStruct, pose):
    # print(pointsStruct.points_relative)
    onesColumn = np.ones((pointsStruct.points_relative.shape[0], 1))
    homogeneousPoints = np.hstack((pointsStruct.points_relative, onesColumn))
    # print(homogeneousPoints)
    # print(pose.shape)
    # print(pose)
    transformedPoints = np.transpose(np.matmul(pose, np.transpose(homogeneousPoints)))


    return PointsForInstance(pointsStruct.scan_num, pointsStruct.instance_num, transformedPoints[:, 0:3], pointsStruct.semantic_label)

def transformPointsSingleScan(pose, points_by_instance):
    return {instance:transformPointsForInstance(pointsStruct, pose) for (instance, pointsStruct) in points_by_instance.items()}

def transformPoints(points_by_scan_by_instance, poses):
    return {scan_num:transformPointsSingleScan(poses[scan_num], points_by_inst) for (scan_num,points_by_inst) in points_by_scan_by_instance.items()}

def combinePointsAcrossScans(points_by_scan_by_instance_map_frame):
    pointsByInstList = {}
    for points_for_scan in points_by_scan_by_instance_map_frame.values():
        for instNum, pointsForInst in points_for_scan.items():
            if (instNum not in pointsByInstList):
                pointsByInstList[instNum] = []
            listsOfPointsForInst = pointsByInstList[instNum]
            listsOfPointsForInst.append(pointsForInst)
            pointsByInstList[instNum] = listsOfPointsForInst

    singleObjPointsForInstNum = {}
    for instNum, pointsListsForInst in pointsByInstList.items():
        semanticClass = pointsListsForInst[0].semantic_label
        allPoints = None
        consistentSemanticClass = True
        initialized = False
        for pointsList in pointsListsForInst:
            if (semanticClass != pointsList.semantic_label):
                print("Semantic classes for instance " + str(instNum) + " are not consistent, skipping")
                consistentSemanticClass = False
                continue
            if (not initialized):
                allPoints = pointsList.points_relative
                initialized = True
            else:
                # print(pointsList.points_relative.shape)
                allPoints = np.vstack((allPoints, pointsList.points_relative))
                # print(allPoints.shape)
        if (consistentSemanticClass):
            # print(allPoints)
            singleObjPointsForInstNum[instNum] = PointsForInstAllScans(instNum, allPoints, semanticClass)
    return singleObjPointsForInstNum

def getCentroidPosition(points):
    return np.mean(points, axis=0)

def getLocalCentroid(points_entries_by_scan_by_instance):
    local_centroids = {}
    for scan_num, points_for_scan in points_entries_by_scan_by_instance.items():
        local_centroids[scan_num] = {instNum:Centroid(instNum, getCentroidPosition(pointsForInst.points_relative), pointsForInst.semantic_label) for instNum, pointsForInst in points_for_scan.items()}
    return local_centroids

def get2DBoundingBoxesRelativeToFrames(pointsByInstThenScan, scanForInst, poses):
      bounding_boxes_2d_for_inst = {}
      for instNum, pointsByScan in pointsByInstThenScan.items():
          initialized = False
          closestScanIndex = scanForInst[instNum]
          # print("Closest index " + str(closestScanIndex))
          closestPoseInv = inv(poses[closestScanIndex])
          pointsTransformedToClosestScanFrame = None
          for scanNum, pointsForScan in pointsByScan.items():
              poseForScan = poses[scanNum]
              transformation = np.matmul(closestPoseInv, poseForScan)
              # print("Closest pose inv")
              # print(closestPoseInv)
              # print(poseForScan)
              # print(transformation)
              # print(pointsForScan.points_relative)
              transformedPoints = transformPointsForInstance(pointsForScan, transformation)
              # print(transformedPoints.points_relative)
              if (not initialized):
                  pointsTransformedToClosestScanFrame = transformedPoints.points_relative
                  initialized = True
              else:
                  pointsTransformedToClosestScanFrame = np.vstack((pointsTransformedToClosestScanFrame, transformedPoints.points_relative))
          if (pointsTransformedToClosestScanFrame.shape[0] >= 4):
              points2D = pointsTransformedToClosestScanFrame[:, 0:2]
              points2DProjectedOnes = np.hstack((points2D, np.ones((points2D.shape[0], 1))))
              points2DProjectedZeros = np.hstack((points2D, np.zeros((points2D.shape[0], 1))))
              points2DProjected = np.vstack((points2DProjectedOnes, points2DProjectedZeros))
              open3DPoints2D = open3d.utility.Vector3dVector(points2DProjected)
              bounding_box = open3d.geometry.OrientedBoundingBox.create_from_points(open3DPoints2D)
          # open3dPoints3D = open3d.utility.Vector3dVector(pointsTransformedToClosestScanFrame)
          # pointCloud = open3d.geometry.PointCloud(open3dPoints3D)

          # input("")
          # open3d.visualization.draw_geometries([bounding_box, pointCloud])
              bounding_box_info = np.asarray(bounding_box.get_box_points())
              bounding_boxes_2d_for_inst[instNum] = (closestScanIndex, bounding_box_info)
      return bounding_boxes_2d_for_inst

def getCenterAndYawForBoundingBoxCorners(bbCorners):
    opposingCorners = [[0, 0], [0, 0]]
    opposingCornersDists = [0, 0]
    for cornerNum in range(bbCorners.shape[0]):
        for otherCornerNum in range(cornerNum + 1, bbCorners.shape[0]):
            if (otherCornerNum == cornerNum):
                continue
            cornersDist = np.linalg.norm(bbCorners[cornerNum, :] - bbCorners[otherCornerNum, :])
            if (cornersDist > opposingCornersDists[0]):
                opposingCornersDists[1] = opposingCornersDists[0]
                opposingCornersDists[0] = cornersDist
                opposingCorners[1] = opposingCorners[0]
                opposingCorners[0] = [cornerNum, otherCornerNum]
            elif (cornersDist > opposingCornersDists[1]):
                opposingCornersDists[1] = cornersDist
                opposingCorners[1] = [cornerNum, otherCornerNum]
    opposingCorner00 = bbCorners[opposingCorners[0][0], :]
    opposingCorner01 = bbCorners[opposingCorners[0][1], :]
    opposingCorner10 = bbCorners[opposingCorners[1][0], :]
    opposingCorner11 = bbCorners[opposingCorners[1][1], :]


    centroidX = (opposingCorner00[0] + opposingCorner01[0]) / 2
    centroidY = (opposingCorner00[1] + opposingCorner01[1]) / 2

    # Long edge is in the x direction
    longEdgeCorners = []
    if (np.linalg.norm(opposingCorner00 - opposingCorner10) > np.linalg.norm(opposingCorner00 - opposingCorner11)):
        longEdgeCorners = [[opposingCorner00, opposingCorner10], [opposingCorner01, opposingCorner11]]
    else:
        longEdgeCorners = [[opposingCorner00, opposingCorner11], [opposingCorner01, opposingCorner10]]

    bRelA = longEdgeCorners[0][1] - longEdgeCorners[0][0]
    angle = math.atan2(bRelA[1], bRelA[0])
    if (angle < 0):
        angle += math.pi

    return (centroidX, centroidY, angle)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./find_object_instances.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
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

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Sequence", FLAGS.sequence)
  print("do_instances", FLAGS.do_instances)
  print("ignore_safety", FLAGS.ignore_safety)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
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

  label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")

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
  if not FLAGS.ignore_safety:
    assert(len(label_names) == len(scan_names))

  color_dict = CFG["color_map"]
  nclasses = len(color_dict)
  scan = SemLaserScan(nclasses, color_dict, project=True)


  extractor = LaserScanPointsExtractor(scan=scan,
                     scan_names=scan_names,
                     label_names=label_names, classes_of_interest=[10])


  output_file = os.path.join(FLAGS.output_dir, "points_by_scan_by_instance_cars_only_" + FLAGS.sequence + ".pkl")
  global_centroids_file = os.path.join(FLAGS.output_dir, "global_centroids_by_instance" + FLAGS.sequence + ".pkl")
  local_centroids_file = os.path.join(FLAGS.output_dir, "local_centroids_by_instance_by_scan" + FLAGS.sequence + ".pkl")

  # run the visualizer
  # points_by_scan_by_instance = extractor.getObjectInstancesForEachScan()
  # joblib.dump(points_by_scan_by_instance, output_file)

  points_by_scan_by_instance = joblib.load(output_file)

  calibration = parse_calibration(os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "calib.txt"))
  # suma_poses = parse_poses(os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "poses.txt"), calibration)
  poses = parse_poses(os.path.join(FLAGS.dataset, "poses", FLAGS.sequence + ".txt"), calibration)
  # print(poses)
  # local_centroids = getLocalCentroid(points_by_scan_by_instance)



  # print(poses)
  # for i in range(len(poses)):
  #     pose = poses[i]
  #     print(i)
  #     print(pose[:, 3])
  #     print(suma_poses[i][:, 3])
  #
  #
  # transformed_points = transformPoints(points_by_scan_by_instance, poses)
  # tranformedPointsByInst = combinePointsAcrossScans(transformed_points)
  #
  # globalCentroids = {instNum:Centroid(instNum, getCentroidPosition(transformed.points_relative), transformed.semantic_label) for instNum, transformed in tranformedPointsByInst.items()}
#   # for instNum, centroid in globalCentroids.items():
#       # print(tranformedPointsByInst[instNum].points_relative)
# #
#       # print(centroid.centroid_point)
#       # print(instNum)
#   joblib.dump(globalCentroids, global_centroids_file)
#   joblib.dump(local_centroids, local_centroids_file)

  csv_parking_spots_3d_out_file = os.path.join(FLAGS.output_dir, "parking_spots_3d_" + FLAGS.sequence + ".csv")

  # for instNum, transformed in tranformedPointsByInst.items():
  #
  #     if (transformed.points_relative.shape[0] >= 4):
  #         open3dPoints = open3d.utility.Vector3dVector(transformed.points_relative)
  #         pointCloud = open3d.geometry.PointCloud(open3dPoints)
  #         bounding_box = open3d.geometry.OrientedBoundingBox.create_from_points(open3dPoints)
  #         print(bounding_box.get_center())
  #         print(np.asarray(bounding_box.get_box_points()))
  #         print(bounding_box.extent)
  #         # open3d.visualization.draw_geometries([bounding_box, pointCloud])
  #     else:
  #         print("Skipping bounding box for inst " + str(instNum))

  # Pseudocode

  pointsByInstThenScan = {}
  scansAtWhichInstIsVisible = {}
  for scanNum, pointsByInst in points_by_scan_by_instance.items():
      for instNum, pointsForInst in pointsByInst.items():
          pointsByScanForInst = {}
          if (instNum in pointsByInstThenScan.keys()):
              pointsByScanForInst = pointsByInstThenScan[instNum]
          pointsByScanForInst[scanNum] = pointsForInst
          pointsByInstThenScan[instNum] = pointsByScanForInst
          instVisible = []
          if (instNum in scansAtWhichInstIsVisible.keys()):
              instVisible = scansAtWhichInstIsVisible[instNum]
          instVisible.append(scanNum)
          scansAtWhichInstIsVisible[instNum] = instVisible

  scansAtWhichInstIsVisibleCsv = os.path.join(FLAGS.output_dir, "scans_where_inst_visible_" + FLAGS.sequence + ".csv")
  with open(scansAtWhichInstIsVisibleCsv, 'w', newline='') as csvfile:
      csvWriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      csvWriter.writerow(['instNum', 'scansWhereVisible'])
      for instNum, visibleScansList in scansAtWhichInstIsVisible.items():
          dataToWrite = [instNum]
          dataToWrite.extend(list(set(visibleScansList)))
          csvWriter.writerow(dataToWrite)


  #
  # closestScanForInst = {}
  # for instNum, centroid in globalCentroids.items():
  #     centroidPosition = centroid.centroid_point
  #     # print(centroidPosition)
  #     closestScan = 0
  #     closestScanDist = float("inf")
  #     for scanNum in range(len(poses)):
  #         pose = poses[scanNum]
  #         position = np.array([pose[0][3], pose[1][3], pose[2][3]])
  #         scanDist = np.linalg.norm(position - centroidPosition)
  #         # print(scanDist)
  #         # print(closestScanDist)
  #         if (scanDist < closestScanDist):
  #             # print("Updating closest scan dist ")
  #             closestScan = scanNum
  #             closestScanDist = scanDist
  #     closestScanForInst[instNum] = closestScan
  #     print("Closest scan for inst " + str(instNum) + ": " + str(closestScan))
  #     pose = poses[closestScan]
  #     print(np.array([pose[0][3], pose[1][3], pose[2][3]]))
  #     print(centroidPosition)
  #
  # print(closestScanForInst)
  #
  # bounding_box_file = os.path.join(FLAGS.output_dir, "bounding_box_2d_" + FLAGS.sequence + ".pkl")
  #
  # # bounding_boxes_2d_for_inst = get2DBoundingBoxesRelativeToFrames(pointsByInstThenScan, closestScanForInst, poses)
  # # joblib.dump(bounding_boxes_2d_for_inst, bounding_box_file)
  # bounding_boxes_2d_for_inst = joblib.load(bounding_box_file)
  # bb_csv_file = os.path.join(FLAGS.output_dir, "global_bounding_box_2d_centroid_" + FLAGS.sequence + ".csv")
  # with open(bb_csv_file, 'w', newline='') as csvfile:
  #     csvWriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  #     csvWriter.writerow(['instNum', 'relativeToScan', 'centroidX', 'centroidY', 'angle'])
  #     for instNum, bbInfo in bounding_boxes_2d_for_inst.items():
  #         scanForBb = bbInfo[0]
  #         bbPoints = bbInfo[1]
  #         # print("Inst num " + str(instNum))
  #         # print(bbPoints)
  #         lastColBbPoints = bbPoints[:, -1]
  #         indicesForZeroPoints = np.argwhere(lastColBbPoints > 0.5)
  #         reducedBBPoints = np.take_along_axis(bbPoints, indicesForZeroPoints, axis=0)[:, 0:2]
  #
  #         (centerX, centerY, angle) =  getCenterAndYawForBoundingBoxCorners(reducedBBPoints)
  #         # print(reducedBBPoints)
  #         # print((centerX, centerY, angle))
  #         # print(reducedBBPoints)
  #         writeContent = [instNum, scanForBb, centerX, centerY, angle]
  #         # writeContent = [instNum, scanForBb]
  #         # writeContent.extend(reducedBBPoints.flatten())
  #         # print(writeContent)
  #         # if (len(writeContent) < 10):
  #         #     print("Inst num " + str(instNum))
  #         #     print("Write conent too short ")
  #         #     print(writeContent)
  #         #     print(bbPoints.shape)
  #         #     print(bbPoints)
  #         #     print(reducedBBPoints.shape)
  #         csvWriter.writerow(writeContent)


  # Get local detections using the accumulation of 10 scans before and after (11 including self)

  # mergeScanCountEachSide = 5
  # localDetections = []
  # for scanNum in points_by_scan_by_instance.keys():
  #     pointsForInstThenScanForQueryScan = {}
  #     for instNum, pointsForInstByScan in pointsByInstThenScan.items():
  #         filteredPointsByScan = {}
  #         for secondScanNum, pointsForScanForInst in pointsForInstByScan.items():
  #             if (abs(secondScanNum - scanNum) <= mergeScanCountEachSide):
  #                 filteredPointsByScan[secondScanNum] = pointsForScanForInst
  #         if (len(filteredPointsByScan) > 0):
  #             pointsForInstThenScanForQueryScan[instNum] = filteredPointsByScan
  #     scanForInst = {instNum:scanNum for instNum in pointsForInstThenScanForQueryScan.keys()}
  #     boundingBoxesForQueryScan = get2DBoundingBoxesRelativeToFrames(pointsForInstThenScanForQueryScan, scanForInst, poses)
  #     for instNum, bbInfo in boundingBoxesForQueryScan.items():
  #         scanForBb = bbInfo[0]
  #         bbPoints = bbInfo[1]
  #         # print("Inst num " + str(instNum))
  #         # print(bbPoints)
  #         lastColBbPoints = bbPoints[:, -1]
  #         indicesForZeroPoints = np.argwhere(lastColBbPoints > 0.5)
  #         reducedBBPoints = np.take_along_axis(bbPoints, indicesForZeroPoints, axis=0)[:, 0:2]
  #
  #         (centerX, centerY, angle) =  getCenterAndYawForBoundingBoxCorners(reducedBBPoints)
  #         localDetections.append([instNum, scanNum, centerX, centerY, angle])
  #
  # localDetectionsCsv = os.path.join(FLAGS.output_dir, "local_bounding_box_2d_centroid_" + FLAGS.sequence + ".csv")
  # with open(localDetectionsCsv, 'w', newline='') as csvfile:
  #     csvWriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  #     csvWriter.writerow(['instNum', 'relativeToScan', 'centroidX', 'centroidY', 'angle'])
  #     for localDetectionsEntry in localDetections:
  #         csvWriter.writerow(localDetectionsEntry)


  # for instNum, transformed in tranformedPointsByInst.items():
  #
  #     print("Min/max")
  #     print(transformed.points_relative.min(axis=0))
  #     print(transformed.points_relative.max(axis=0))
  #     # print("Diff ")
  #     diff = np.abs(transformed.points_relative.min(axis=0) - transformed.points_relative.max(axis=0))
  #     if (diff.max() > 7):
  #         print("Instance " + str(instNum))
  #         print("Diff greater than 7")
  #         print(diff)
  #         x_s = transformed.points_relative[:, 0]
  #         y_s = transformed.points_relative[:, 1]
  #         z_s = transformed.points_relative[:, 2]
  #         x_histo = np.histogram(x_s)
  #         y_histo = np.histogram(y_s)
  #         z_histo = np.histogram(z_s)
  #         print(x_histo)
  #         print(y_histo)
  #         print(z_histo)
  #         plt.hist(x_s, bins=15)
  #         plt.show()
  #         plt.hist(y_s, bins=15)
  #         plt.show()
  #         plt.hist(z_s, bins=15)
  #         plt.legend()
  #         plt.show()
  #         plt.hist(y_histo, bins='auto')
  #         plt.show()
  #         plt.hist(z_histo, bins='auto')
  #         plt.show()
