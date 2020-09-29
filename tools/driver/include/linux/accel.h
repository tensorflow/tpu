/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef __ACCEL_H__
#define __ACCEL_H__ 
#include <linux/module.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/types.h>
#include <linux/statfs.h>
#include <linux/kobject.h>
#define ACCEL_MAJOR 121
#define ACCEL_BLOCK_MAJOR 121
struct accel_dev;
struct accel_dev {
 const char *accel_type;
 const char *chip_vendor;
 const char *chip_model;
 const char *chip_revision;
 unsigned int fw_version[4];
 const char *logic_vendor;
 const char *logic_model;
 const char *logic_revision;
 const char *logic_build_cl;
 const char *logic_build_time;
 const struct file_operations *fops;
 int id;
 struct device dev;
 struct cdev cdev;
 void (*release)(struct accel_dev *);
 const char *state;
 struct kobject *physical_functions;
 struct kset *scalar_resources;
};
static inline const char *accel_dev_name(const struct accel_dev *adev)
{
 return dev_name(&adev->dev);
}
static inline struct accel_dev *accel_dev_get(struct accel_dev *adev)
{
 if (adev)
  get_device(&adev->dev);
 return adev;
}
static inline void accel_dev_put(struct accel_dev *adev)
{
 if (adev)
  put_device(&adev->dev);
}
static inline void accel_dev_set_state(struct accel_dev *adev,
           const char *state)
{
 adev->state = state;
}
static inline const char *accel_dev_get_state(struct accel_dev *adev)
{
 if (WARN_ON(!adev))
  return NULL;
 return adev->state;
}
extern int accel_dev_init(struct accel_dev *dev, struct device *parent,
     void (*release)(struct accel_dev *));
extern int accel_dev_register(struct accel_dev *adev);
extern void accel_dev_unregister(struct accel_dev *adev);
extern struct accel_dev *accel_dev_get_by_devt(dev_t devt);
extern struct accel_dev *accel_dev_get_by_parent(struct device *parent);
#define to_accel_dev(dev) ((struct accel_dev *)dev_get_drvdata(dev))
struct accel_scalar_resource {
 struct kobject kobj;
 struct kobject *functions;
 struct kobject *iommu_groups;
};
#define to_accel_scalar_resource(x) \
 container_of(x, struct accel_scalar_resource, kobj)
extern int accel_add_physical_function(struct accel_dev *adev,
    struct device *dev);
extern void accel_remove_physical_function(struct accel_dev *adev,
     struct device *dev);
extern struct accel_scalar_resource *accel_scalar_resource_add(
 struct accel_dev *adev, int idx);
extern void accel_scalar_resource_remove(
 struct accel_scalar_resource *resource);
extern int accel_scalar_resource_add_function(
 struct accel_scalar_resource *resource,
 struct device *dev);
extern void accel_scalar_resource_remove_function(
 struct accel_scalar_resource *resource,
 struct device *dev);
extern int accel_scalar_resource_add_iommu_group(
 struct accel_scalar_resource *resource,
 struct device *dev);
extern int accel_scalar_resource_remove_iommu_group(
 struct accel_scalar_resource *resource,
 struct device *dev);
#endif
