/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/accel.h>
#include <linux/fs.h>
#include <linux/kdev_t.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 15, 0)
#include <linux/genhd.h>
#else
#include <linux/blkdev.h>
#endif
#define ACCEL_MAX_DEVICES 512
static bool accel_class_registered;
static struct class accel_class;
static DECLARE_BITMAP(accel_id_bitmap, ACCEL_MAX_DEVICES);
static int accel_alloc_id(void)
{
 int id;
 do {
  id = find_first_zero_bit(accel_id_bitmap, ACCEL_MAX_DEVICES);
  if (id >= ACCEL_MAX_DEVICES)
   return -ENOSPC;
 } while (test_and_set_bit(id, accel_id_bitmap));
 return id;
}
static void accel_free_id(int id)
{
 if (id >= 0)
  clear_bit(id, accel_id_bitmap);
}
static int accel_dev_chrdev_register(struct accel_dev *adev)
{
 int ret;
 if (!adev->fops)
  return 0;
 adev->dev.devt = MKDEV(ACCEL_MAJOR, adev->id);
 cdev_init(&adev->cdev, adev->fops);
 ret = cdev_add(&adev->cdev, adev->dev.devt, 1);
 if (ret) {
  dev_err(&adev->dev, "Error adding chrdev");
  goto err;
 }
 dev_dbg(&adev->dev, "chrdev added");
 return 0;
err:
 adev->dev.devt = 0;
 return ret;
}
static void accel_dev_chrdev_unregister(struct accel_dev *adev)
{
 if (!adev->fops)
  return;
 cdev_del(&adev->cdev);
 dev_dbg(&adev->dev, "chrdev removed");
}
static void accel_dev_release(struct device *dev)
{
 struct accel_dev *adev = to_accel_dev(dev);
 accel_free_id(adev->id);
 adev->release(adev);
}
int accel_dev_init(struct accel_dev *adev,
     struct device *parent,
     void (*release)(struct accel_dev *))
{
 int id;
 if (WARN_ON(!adev || !release))
  return -EINVAL;
 adev->id = -1;
 adev->dev.class = &accel_class;
 adev->dev.parent = parent;
 adev->dev.release = accel_dev_release;
 adev->release = release;
 dev_set_drvdata(&adev->dev, adev);
 device_initialize(&adev->dev);
 adev->state = "initializing";
 if (!accel_class_registered)
  return -EAGAIN;
 id = accel_alloc_id();
 if (id < 0)
  return id;
 adev->id = id;
 return dev_set_name(&adev->dev, "accel%d", id);
}
EXPORT_SYMBOL(accel_dev_init);
int accel_dev_register(struct accel_dev *adev)
{
 int ret;
 if (!adev)
  return -EINVAL;
 ret = accel_dev_chrdev_register(adev);
 if (ret) {
  dev_err(&adev->dev, "accel_dev chrdev register failed");
  return ret;
 }
 ret = device_add(&adev->dev);
 if (ret) {
  dev_err(&adev->dev, "accel_dev device add failed");
  goto out_chrdev_unregister;
 }
 adev->physical_functions =
     kobject_create_and_add("physical_functions", &adev->dev.kobj);
 adev->scalar_resources =
     kset_create_and_add("resources", NULL, &adev->dev.kobj);
 dev_dbg(&adev->dev, "accel_dev registered");
 return 0;
out_chrdev_unregister:
 accel_dev_chrdev_unregister(adev);
 return ret;
}
EXPORT_SYMBOL(accel_dev_register);
void accel_dev_unregister(struct accel_dev *adev)
{
 if (!adev)
  return;
 kset_unregister(adev->scalar_resources);
 kobject_put(adev->physical_functions);
 accel_dev_chrdev_unregister(adev);
 dev_dbg(&adev->dev, "accel_dev unregistering");
 device_del(&adev->dev);
}
EXPORT_SYMBOL(accel_dev_unregister);
static int accel_dev_match_devt(struct device *dev, const void *data)
{
 const dev_t *devt = data;
 return dev->devt == *devt;
}
struct accel_dev *accel_dev_get_by_devt(dev_t devt)
{
 struct device *dev;
 dev = class_find_device(
   &accel_class, NULL, &devt, accel_dev_match_devt);
 if (!dev)
  return NULL;
 return to_accel_dev(dev);
}
EXPORT_SYMBOL(accel_dev_get_by_devt);
static int accel_dev_match_parent(struct device *dev, const void *data)
{
 const struct device *parent = data;
 return dev->parent == parent;
}
struct accel_dev *accel_dev_get_by_parent(struct device *parent)
{
 struct device *dev;
 dev = class_find_device(&accel_class, NULL,
   parent, accel_dev_match_parent);
 if (!dev)
  return NULL;
 return to_accel_dev(dev);
}
EXPORT_SYMBOL(accel_dev_get_by_parent);
static ssize_t state_show(struct device *dev,
     struct device_attribute *attr,
     char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 return sprintf(buf, "%s\n", adev->state);
}
static DEVICE_ATTR_RO(state);
static ssize_t accel_type_show(struct device *dev,
          struct device_attribute *attr,
          char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 return sprintf(buf, "%s\n", adev->accel_type);
}
static DEVICE_ATTR_RO(accel_type);
static ssize_t chip_vendor_show(struct device *dev,
    struct device_attribute *attr,
    char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->chip_vendor;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(chip_vendor);
static ssize_t chip_model_show(struct device *dev,
          struct device_attribute *attr,
          char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->chip_model;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(chip_model);
static ssize_t chip_revision_show(struct device *dev,
      struct device_attribute *attr,
      char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->chip_revision;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(chip_revision);
static ssize_t chip_serial_number_show(struct device *dev,
           struct device_attribute *attr,
           char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->chip_serial_number;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(chip_serial_number);
static ssize_t fw_version_show(struct device *dev,
          struct device_attribute *attr,
          char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 if (adev->fw_version_str) {
  return sprintf(buf, "%s\n", adev->fw_version_str);
 } else {
  return sprintf(buf, "%u.%u.%u.%u\n",
          adev->fw_version[0], adev->fw_version[1],
          adev->fw_version[2], adev->fw_version[3]);
 }
}
static DEVICE_ATTR_RO(fw_version);
static ssize_t logic_vendor_show(struct device *dev,
     struct device_attribute *attr,
     char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->logic_vendor;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(logic_vendor);
static ssize_t logic_model_show(struct device *dev,
    struct device_attribute *attr,
    char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->logic_model;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(logic_model);
static ssize_t logic_revision_show(struct device *dev,
       struct device_attribute *attr,
       char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->logic_revision;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(logic_revision);
static ssize_t logic_build_cl_show(struct device *dev,
       struct device_attribute *attr,
       char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->logic_build_cl;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(logic_build_cl);
static ssize_t logic_build_time_show(struct device *dev,
         struct device_attribute *attr,
         char *buf)
{
 struct accel_dev *adev = to_accel_dev(dev);
 const char *v = adev->logic_build_time;
 if (!v)
  v = "";
 return sprintf(buf, "%s\n", v);
}
static DEVICE_ATTR_RO(logic_build_time);
static struct attribute *accel_dev_attrs[] = {
 &dev_attr_accel_type.attr,
 &dev_attr_chip_vendor.attr,
 &dev_attr_chip_model.attr,
 &dev_attr_chip_revision.attr,
 &dev_attr_chip_serial_number.attr,
 &dev_attr_fw_version.attr,
 &dev_attr_logic_vendor.attr,
 &dev_attr_logic_model.attr,
 &dev_attr_logic_revision.attr,
 &dev_attr_logic_build_cl.attr,
 &dev_attr_logic_build_time.attr,
 &dev_attr_state.attr,
 NULL,
};
ATTRIBUTE_GROUPS(accel_dev);
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 2, 0)
static int accel_dev_uevent(struct device *dev, struct kobj_uevent_env *env)
#else
static int accel_dev_uevent(const struct device *dev, struct kobj_uevent_env *env)
#endif
{
 struct accel_dev *adev = to_accel_dev(dev);
 int retval = 0;
 if (adev->accel_type)
  retval = add_uevent_var(env, "ACCEL_TYPE=%s",
     adev->accel_type);
 if (retval)
  goto bail_out;
 if (adev->chip_vendor)
  retval = add_uevent_var(env, "ACCEL_CHIP_VENDOR=%s",
     adev->chip_vendor);
 if (retval)
  goto bail_out;
 if (adev->chip_model)
  retval = add_uevent_var(env, "ACCEL_CHIP_MODEL=%s",
     adev->chip_model);
 if (retval)
  goto bail_out;
 if (adev->logic_vendor)
  retval = add_uevent_var(env, "ACCEL_LOGIC_VENDOR=%s",
     adev->logic_vendor);
 if (retval)
  goto bail_out;
 if (adev->logic_model)
  retval = add_uevent_var(env, "ACCEL_LOGIC_MODEL=%s",
     adev->logic_model);
bail_out:
 return retval;
}
static struct class accel_class = {
 .name = "accel",
 .owner = THIS_MODULE,
 .dev_groups = accel_dev_groups,
 .dev_uevent = accel_dev_uevent,
};
int accel_add_physical_function(struct accel_dev *adev, struct device *dev)
{
 if (WARN_ON(adev == NULL || dev == NULL))
  return -EINVAL;
 return sysfs_create_link(adev->physical_functions,
    &dev->kobj, dev_name(dev));
}
EXPORT_SYMBOL(accel_add_physical_function);
void accel_remove_physical_function(struct accel_dev *adev, struct device *dev)
{
 if (WARN_ON(adev == NULL || dev == NULL))
  return;
 sysfs_remove_link(adev->physical_functions, dev_name(dev));
}
EXPORT_SYMBOL(accel_remove_physical_function);
static int __init accel_init(void)
{
 int ret;
 dev_t chr_region = MKDEV(ACCEL_MAJOR, 0);
 ret = register_chrdev_region(chr_region, ACCEL_MAX_DEVICES, "accel");
 if (ret < 0) {
  pr_err("accel: cannot register char major number %d\n",
         ACCEL_MAJOR);
  return ret;
 }
 ret = register_blkdev(ACCEL_BLOCK_MAJOR, "accel_block");
 if (ret < 0) {
  pr_err("accel: cannot register block major number %d\n",
         ACCEL_BLOCK_MAJOR);
  goto out_chr_region_free;
 }
 ret = class_register(&accel_class);
 if (ret) {
  pr_err("accel: class_register failed for accel\n");
  goto out_blk_region_free;
 }
 accel_class_registered = true;
 pr_info("accel: class registered successfully (Major no = %d)\n",
  ACCEL_MAJOR);
 return 0;
out_blk_region_free:
 unregister_blkdev(ACCEL_BLOCK_MAJOR, "accel_block");
out_chr_region_free:
 unregister_chrdev_region(chr_region, ACCEL_MAX_DEVICES);
 return ret;
}
static void __exit accel_exit(void)
{
 class_unregister(&accel_class);
 unregister_blkdev(ACCEL_BLOCK_MAJOR, "accel_block");
 unregister_chrdev_region(MKDEV(ACCEL_MAJOR, 0), ACCEL_MAX_DEVICES);
}
module_init(accel_init)
module_exit(accel_exit)
MODULE_DESCRIPTION("Google Accelerator Class driver");
MODULE_AUTHOR("Googler <noreply@google.com>");
MODULE_AUTHOR("Googler <noreply@google.com>");
MODULE_LICENSE("GPL");
