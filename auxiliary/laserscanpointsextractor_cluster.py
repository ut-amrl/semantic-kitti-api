#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
from matplotlib import pyplot as plt
from auxiliary.laserscan import LaserScan, SemLaserScan

class PointsForInstance:

    def __init__(self, scan_num, instance_num, points_relative, semantic_label):
        self.scan_num = scan_num
        self.instance_num = instance_num
        self.points_relative = points_relative
        self.semantic_label = semantic_label




class LaserScanPointsExtractorCluster :

  def __init__(self, scan, scan_names, label_names, classes_of_interest):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.total = len(self.scan_names)
    self.offset = 0
    self.classes_of_interest = classes_of_interest

    # self.reset()
    self.update_scan()

  def update_scan(self):
    # first open data

    scan_number = self.offset
    # print("Scan num")
    # print(scan_number)
    self.scan.open_scan(self.scan_names[scan_number])
    self.scan.open_label(self.label_names[scan_number])

    inst_labels = self.scan.label
    semantic_labels = self.scan.sem_label
    unique_inst_labels = np.unique(inst_labels)
    # print("Unique inst labels")
    # print(unique_inst_labels)

    semantic_label_for_inst = {}

    instance_points = {}
    points_for_scan = self.scan.points
    # print("Points for scan shape")
    # print(self.scan.points.shape)
    remissions = self.scan.remissions
    # print(remissions.shape)
    remissions_2d = np.reshape(remissions, (remissions.shape[0], 1))
    # print(remissions_2d.shape)
    joined = np.concatenate((points_for_scan, remissions_2d), axis=1)
    # print(joined.shape)
    # print(points_for_scan[-2, :])
    # print(remissions[-2])
    # print(joined[-2, :])


    for inst_label in unique_inst_labels:
    # for inst_label in [6]:
    #     if (inst_label == 0):
            # print("Skipping unlabeled")
            # continue
        # print("Inst label? ")
        # print(inst_label)
        arr = inst_labels
        indices_for_inst = np.argwhere(inst_label == arr)

        # points_for_inst = np.take_along_axis(points_for_scan, indices_for_inst, axis=0)
        points_for_inst= np.take_along_axis(joined, indices_for_inst, axis=0)
        # print("Points shape")
        # print(points_for_inst.shape)
        # print(indices_for_inst.shape)

        # print(np.take(inst_labels, indices_for_inst))
        # print(indices_for_inst.shape)
        # print(semantic_labels.shape)
        # print(inst_labels.shape)
        # print(np.take(semantic_labels, indices_for_inst))
        semantic_labels_not_unique = np.take(semantic_labels, indices_for_inst)
        semantic_labels_for_inst = np.unique(semantic_labels_not_unique)
        # print(semantic_labels_for_inst)
        if (semantic_labels_for_inst.size > 1):
            print("Error! Multiple semantic labels for same inst")
            print(semantic_labels_for_inst)
            print(np.unique(np.take(inst_labels, indices_for_inst)))

        # if (semantic_labels_for_inst.size == 0):
            # continue
        semantic_label_for_inst = semantic_labels_for_inst[0]
        # print("Semantic Label for inst")
        # print(semantic_label_for_inst)

        if (semantic_label_for_inst in self.classes_of_interest):
            # print("Including points")
            instance_points[inst_label] = PointsForInstance(scan_number, inst_label, points_for_inst, semantic_label_for_inst)
            # input("")
            instance_only_label = inst_label >> 16
            # if (instance_only_label == 0):
            #     print("Warning: instance for semantic class " + str(semantic_label_for_inst) + " was 0")
                # input("")

    # print(instance_points)
    # input("")
    return instance_points

  def getObjectInstancesForEachScan(self):
      instances_for_scans = {}
      while (self.offset < self.total):
          instances_for_scans[self.offset] = self.update_scan()
          self.offset += 1
      print("Done!")
      return instances_for_scans
          # if self.offset >= self.total:
            # self.offset = 0
